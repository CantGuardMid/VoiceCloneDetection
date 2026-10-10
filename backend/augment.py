"""
augment.py
----------
Generates "pseudo-synthetic" audio from a real/clean clip by applying
signal-processing degradations known to correlate with TTS/voice-conversion
vocoder output:

  1. Frequency-axis smoothing of the magnitude spectrum -> loses the fine
     harmonic/formant detail a real vocal tract produces.
  2. Griffin-Lim style phase re-estimation -> replaces the natural phase
     with an algorithmically reconstructed one, mimicking vocoders that
     do not model phase directly.
  3. Mild bit-depth quantization -> mimics neural-vocoder quantization noise.
  4. Envelope flattening -> reduces natural shimmer (amplitude micro-variation).

IMPORTANT / HONESTY NOTE:
This is a proxy task, not real deepfake audio. There is no internet access
in this environment to download a labeled corpus (e.g. ASVspoof) or a
pretrained vocoder, so we cannot train against *actual* TTS output here.
This module lets the bundled classifier learn to key off the same class of
spectral/phase artifacts that real anti-spoofing research (ASVspoof, AASIST)
targets, so the pipeline is genuinely trained and functional end-to-end.
Before any real deployment, retrain `model.py` on ASVspoof/WaveFake with
actual TTS/voice-conversion samples.
"""

import numpy as np
from scipy.signal import stft, istft


def _smooth_freq_axis(mag: np.ndarray, kernel: int = 5) -> np.ndarray:
    kernel = max(kernel, 1)
    pad = kernel // 2
    padded = np.pad(mag, ((0, 0), (pad, pad)), mode="edge")
    kernel_arr = np.ones(kernel) / kernel
    return np.apply_along_axis(lambda m: np.convolve(m, kernel_arr, mode="valid"), 1, padded)


def _griffin_lim(mag: np.ndarray, sr: int, nperseg: int, noverlap: int, n_iter: int = 12) -> np.ndarray:
    angles = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=mag.shape))
    spec = mag * angles
    for _ in range(n_iter):
        _, rec = istft(spec, fs=sr, nperseg=nperseg, noverlap=noverlap)
        _, _, re_spec = stft(rec, fs=sr, nperseg=nperseg, noverlap=noverlap)
        angles = np.exp(1j * np.angle(re_spec))
        # re_spec may have one extra frame vs mag due to framing edge effects
        n = min(mag.shape[1], angles.shape[1])
        spec = mag[:, :n] * angles[:, :n]
    _, rec = istft(spec, fs=sr, nperseg=nperseg, noverlap=noverlap)
    return rec


def _quantize(samples: np.ndarray, bits: int = 8) -> np.ndarray:
    levels = 2 ** bits
    q = np.round(samples * (levels / 2)) / (levels / 2)
    return np.clip(q, -1.0, 1.0)


def _flatten_envelope(samples: np.ndarray, sr: int, strength: float = 0.5) -> np.ndarray:
    win = max(int(sr * 0.02), 1)
    env = np.abs(samples)
    kernel = np.ones(win) / win
    env_smooth = np.convolve(env, kernel, mode="same") + 1e-6
    target = np.mean(env_smooth)
    gain = (target / env_smooth) ** strength
    return samples * gain


def degrade_audio(samples: np.ndarray, sr: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    samples = samples.astype(np.float64)
    nperseg = 512
    noverlap = nperseg // 2

    _, _, spec = stft(samples, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(spec)
    mag_smooth = _smooth_freq_axis(mag, kernel=5)

    rec = _griffin_lim(mag_smooth, sr, nperseg, noverlap, n_iter=10)

    # length can drift slightly through stft/istft round trips - fix it
    if len(rec) < len(samples):
        rec = np.pad(rec, (0, len(samples) - len(rec)))
    else:
        rec = rec[: len(samples)]

    rec = _flatten_envelope(rec, sr, strength=0.4)
    peak = np.max(np.abs(rec)) + 1e-9
    rec = rec / peak * 0.9
    rec = _quantize(rec, bits=8)

    return rec.astype(np.float32)
