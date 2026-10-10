"""
retrain.py
----------
Retrain the synthetic-speech classifier on REAL labeled audio instead of
the self-supervised bootstrap data in model.py.

Usage:
    python retrain.py --real data/real_voices --fake data/fake_voices

Expected folder layout (any nesting is fine, it just globs *.wav):
    data/real_voices/*.wav     <- genuine human speech clips
    data/fake_voices/*.wav     <- AI-generated / cloned / voice-converted clips

Each file is split into the same overlapping 1.5s windows used at inference
time (audio_features.chunk_windows), so a 5-second clip yields several
training examples and the model sees the same chunk statistics it will be
scored on live. Run with --test-size 0 to train on everything (small
datasets); otherwise it holds out a stratified test split and prints a
classification report.

On success this overwrites model_store/synthetic_detector.pkl -- restart
app.py (or it'll pick it up fresh next run) to serve the retrained model.
"""

import os
import glob
import argparse
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from audio_features import extract_feature_vector, chunk_windows, resample_linear, read_wav_bytes, SR_TARGET
from model import SyntheticSpeechDetector, MODEL_PATH, MODEL_DIR


def load_wav_file(path):
    with open(path, "rb") as f:
        samples, sr = read_wav_bytes(f.read())
    if sr != SR_TARGET:
        samples = resample_linear(samples, sr, SR_TARGET)
        sr = SR_TARGET
    return samples, sr


def collect_features(folder, label, verbose=True):
    paths = sorted(glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True))
    if not paths:
        print(f"  WARNING: no .wav files found under {folder}")
    X, meta = [], []
    for path in paths:
        try:
            samples, sr = load_wav_file(path)
        except Exception as e:
            print(f"  skipping {path}: could not read ({e})")
            continue
        n_chunks = 0
        for start_sec, end_sec, chunk in chunk_windows(samples, sr):
            fv, _, _ = extract_feature_vector(chunk, sr)
            X.append(fv)
            meta.append({"file": os.path.basename(path), "start": start_sec, "end": end_sec})
            n_chunks += 1
        if verbose:
            print(f"  {os.path.basename(path)}: {n_chunks} window(s)")
    y = [label] * len(X)
    return X, y, meta


def main():
    parser = argparse.ArgumentParser(description="Retrain the synthetic-speech detector on real labeled audio.")
    parser.add_argument("--real", required=True, help="Folder of genuine human-voice .wav files")
    parser.add_argument("--fake", required=True, help="Folder of AI-generated/cloned .wav files")
    parser.add_argument("--test-size", type=float, default=0.25, help="Held-out fraction for evaluation (0 = train on everything)")
    parser.add_argument("--out", default=MODEL_PATH, help="Where to save the trained model")
    args = parser.parse_args()

    print(f"Scanning real voices in: {args.real}")
    X_real, y_real, meta_real = collect_features(args.real, label=0)
    print(f"Scanning fake/synthetic voices in: {args.fake}")
    X_fake, y_fake, meta_fake = collect_features(args.fake, label=1)

    X = np.array(X_real + X_fake)
    y = np.array(y_real + y_fake)

    if len(X) == 0:
        print("\nNo usable audio found in either folder. Aborting.")
        return
    if len(set(y.tolist())) < 2:
        print("\nNeed at least one example in BOTH folders to train a classifier. Aborting.")
        return

    print(f"\nTotal training windows: {len(X)}  (real={len(X_real)}, fake={len(X_fake)})")

    det = SyntheticSpeechDetector()

    if args.test_size and args.test_size > 0 and len(X) >= 8:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        det.fit(X_train, y_train)
        y_pred = [1 if det.predict_proba(x) > 0.5 else 0 for x in X_test]
        print("\n--- Held-out evaluation ---")
        print(classification_report(y_test, y_pred, target_names=["real", "synthetic"], zero_division=0))
        print("Confusion matrix ([[TN, FP], [FN, TP]]):")
        print(confusion_matrix(y_test, y_pred))
        # refit on everything for the final saved model
        det.fit(X, y)
    else:
        print("\nSkipping held-out split (dataset too small or --test-size 0) - training on all data.")
        det.fit(X, y)

    det.save(args.out)

    info = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "real_windows": len(X_real),
        "fake_windows": len(X_fake),
        "real_files": sorted(set(m["file"] for m in meta_real)),
        "fake_files": sorted(set(m["file"] for m in meta_fake)),
    }
    with open(os.path.join(MODEL_DIR, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nSaved retrained model to: {args.out}")
    print("Restart backend/app.py (or it'll load fresh next launch) to serve it.")


if __name__ == "__main__":
    main()
