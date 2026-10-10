"""
model.py
--------
Trains and serves the "synthetic speech" classifier.

Because this prototype runs fully offline (no internet -> no ASVspoof/
WaveFake download, no pretrained AASIST/ECAPA-TDNN weights), we bootstrap a
small labeled dataset ourselves:

  1. Synthesize several "clean voice-like" signals -- additive harmonic
     synthesis with natural micro-jitter, shaped by resonant (formant-like)
     filters, at a handful of different pitches -- standing in for
     real human speech statistics.
  2. Degrade each with augment.degrade_audio() to produce a paired
     "pseudo-synthetic" version (see augment.py for what that simulates).
  3. Extract the same acoustic feature vector from both.
  4. Train a small MLP classifier (scikit-learn) to separate real (0) vs
     pseudo-synthetic (1).

This gives a genuinely trained, working classifier for the demo. It is a
PROXY for the real task -- swap this bootstrap step for training on
ASVspoof/WaveFake (and consider a CNN over raw mel-spectrograms, e.g.
AASIST/RawNet2, via PyTorch) before using this on real deepfake audio.
"""

import os
import pickle
import numpy as np
from scipy.signal import lfilter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from audio_features import extract_feature_vector, SR_TARGET
from augment import degrade_audio

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model_store")
MODEL_PATH = os.path.join(MODEL_DIR, "synthetic_detector.pkl")


def _formant_filter(signal: np.ndarray, sr: int, center_freqs, bandwidths):
    out = np.zeros_like(signal)
    for f_c, bw in zip(center_freqs, bandwidths):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f_c / sr
        b = [1 - r]
        a = [1, -2 * r * np.cos(theta), r ** 2]
        out += lfilter(b, a, signal)
    return out / max(len(center_freqs), 1)


def _synth_voice(sr: int, duration: float, f0_base: float, formants, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n = int(sr * duration)
    t = np.arange(n) / sr

    jitter_noise = rng.normal(0, 1, n)
    jitter_noise = np.convolve(jitter_noise, np.ones(200) / 200, mode="same")
    f0 = f0_base + 3.0 * np.sin(2 * np.pi * 0.4 * t) + 1.5 * jitter_noise

    phase = 2 * np.pi * np.cumsum(f0) / sr
    harmonics = sum(
        (1.0 / k) * np.sin(k * phase) for k in range(1, 6)
    )

    shimmer = 1.0 + 0.03 * rng.normal(0, 1, n)
    shimmer = np.convolve(shimmer, np.ones(80) / 80, mode="same")
    excitation = harmonics * shimmer

    voiced = _formant_filter(excitation, sr, formants["freqs"], formants["bws"])
    voiced += 0.01 * rng.normal(0, 1, n)  # breath noise
    voiced = voiced / (np.max(np.abs(voiced)) + 1e-9) * 0.85
    return voiced.astype(np.float32)


# a handful of distinct "speaker/vowel" configs for bootstrap diversity
_BOOTSTRAP_CONFIGS = [
    dict(f0_base=110, formants=dict(freqs=[700, 1200, 2600], bws=[80, 90, 120])),
    dict(f0_base=140, formants=dict(freqs=[500, 1700, 2500], bws=[70, 100, 130])),
    dict(f0_base=95,  formants=dict(freqs=[650, 1100, 2400], bws=[90, 90, 110])),
    dict(f0_base=180, formants=dict(freqs=[550, 2000, 2800], bws=[60, 110, 140])),
    dict(f0_base=125, formants=dict(freqs=[750, 1500, 2700], bws=[85, 95, 125])),
    dict(f0_base=160, formants=dict(freqs=[600, 1800, 2600], bws=[75, 105, 120])),
]


def build_bootstrap_dataset(sr=SR_TARGET, duration=1.5):
    X, y = [], []
    for i, cfg in enumerate(_BOOTSTRAP_CONFIGS):
        for rep in range(3):  # a few seeded variations per config
            seed = i * 10 + rep
            clean = _synth_voice(sr, duration, cfg["f0_base"], cfg["formants"], seed=seed)
            fake = degrade_audio(clean, sr, seed=seed)

            fv_clean, _, _ = extract_feature_vector(clean, sr)
            fv_fake, _, _ = extract_feature_vector(fake, sr)

            X.append(fv_clean); y.append(0)
            X.append(fv_fake); y.append(1)
    return np.array(X), np.array(y)


class SyntheticSpeechDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=2000,
            random_state=42,
        )
        self.trained = False

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        self.trained = True

    def predict_proba(self, feature_vector: np.ndarray) -> float:
        if not self.trained:
            raise RuntimeError("Model not trained/loaded yet")
        Xs = self.scaler.transform(feature_vector.reshape(1, -1))
        proba = self.clf.predict_proba(Xs)[0]
        classes = list(self.clf.classes_)
        return float(proba[classes.index(1)])  # P(synthetic)

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "clf": self.clf}, f)

    @classmethod
    def load_or_train(cls, path=MODEL_PATH):
        inst = cls()
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            inst.scaler = data["scaler"]
            inst.clf = data["clf"]
            inst.trained = True
            return inst
        X, y = build_bootstrap_dataset()
        inst.fit(X, y)
        inst.save(path)
        return inst


if __name__ == "__main__":
    print("Training bootstrap synthetic-speech detector...")
    X, y = build_bootstrap_dataset()
    print("Dataset:", X.shape, "positives:", int(y.sum()), "/", len(y))
    det = SyntheticSpeechDetector()
    det.fit(X, y)
    det.save()
    print("Saved to", MODEL_PATH)

    # quick holdout sanity check on fresh synth samples (not perfect science,
    # just a smoke test that training actually learned something)
    Xtest, ytest = build_bootstrap_dataset(duration=1.2)
    preds = [1 if det.predict_proba(x) > 0.5 else 0 for x in Xtest]
    acc = np.mean(np.array(preds) == ytest)
    print(f"Self-consistency accuracy on fresh bootstrap samples: {acc:.2%}")
