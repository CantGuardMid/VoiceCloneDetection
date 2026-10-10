"""
app.py
------
Flask backend for the Voice Clone Detection prototype.

Endpoints:
  GET  /                    -> frontend
  POST /api/enroll          -> upload a reference clip of the speaker
  POST /api/analyze         -> upload a "call" clip; chunked analysis
  POST /api/text-check      -> paste a transcript; AI-phrasing heuristic score
  POST /api/reset           -> clear enrolled speaker (demo convenience)
  GET  /api/health          -> status

Only WAV (PCM, 16-bit or 32-bit float, mono or stereo) is supported --
the frontend records audio in-browser and encodes it to 16-bit PCM WAV
itself, so no server-side decoding of compressed formats (mp3/webm) is
needed. See frontend/static/recorder.js.
"""

import io
import os
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audio_features import (
    extract_feature_vector, cosine_similarity, SR_TARGET, chunk_windows, read_wav_bytes,
)
from model import SyntheticSpeechDetector
from text_authenticity import analyze_text

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(
    __name__,
    static_folder=os.path.join(FRONTEND_DIR, "static"),
    static_url_path="/static",
)

DETECTOR = SyntheticSpeechDetector.load_or_train()

# in-memory "enrollment store" -- fine for a single-user demo.
STATE = {"enrolled_embedding": None}

# ------------------------------------------------------------------ WAV IO
def read_wav_from_filestorage(file_storage) -> (np.ndarray, int):
    return read_wav_bytes(file_storage.read())


def make_melspec_png_b64(log_mel: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=110)
    ax.imshow(log_mel.T, aspect="auto", origin="lower", cmap="mako" if "mako" in plt.colormaps() else "viridis")
    ax.set_xlabel("Frame", fontsize=8)
    ax.set_ylabel("Mel bin", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#0D1B2A")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def fuse_risk(synthetic_prob: float, speaker_similarity):
    """Combine synthetic-speech probability + speaker-verification similarity
    into one risk score in [0, 1]. speaker_similarity is None if no speaker
    has been enrolled yet."""
    if speaker_similarity is None:
        risk = synthetic_prob
    else:
        mismatch = 1.0 - max(speaker_similarity, 0.0)
        risk = 0.6 * synthetic_prob + 0.4 * mismatch
    return float(np.clip(risk, 0.0, 1.0))


def risk_label(risk: float) -> str:
    if risk < 0.35:
        return "SAFE"
    elif risk < 0.65:
        return "SUSPICIOUS"
    return "ALERT"


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "enrolled": STATE["enrolled_embedding"] is not None})


@app.route("/api/enroll", methods=["POST"])
def enroll():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file uploaded (field name 'audio')"}), 400
    try:
        samples, sr = read_wav_from_filestorage(request.files["audio"])
    except Exception as e:
        return jsonify({"error": f"could not read WAV file: {e}"}), 400

    if len(samples) / sr < 0.6:
        return jsonify({"error": "reference clip too short - record at least 1-2 seconds"}), 400

    _, embedding, extras = extract_feature_vector(samples, sr)
    STATE["enrolled_embedding"] = embedding.tolist()

    return jsonify({
        "status": "enrolled",
        "duration_sec": round(len(samples) / sr, 2),
        "pitch_hz": round(extras["f0_mean"], 1),
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    STATE["enrolled_embedding"] = None
    return jsonify({"status": "reset"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file uploaded (field name 'audio')"}), 400
    try:
        samples, sr = read_wav_from_filestorage(request.files["audio"])
    except Exception as e:
        return jsonify({"error": f"could not read WAV file: {e}"}), 400

    if sr != SR_TARGET:
        from audio_features import resample_linear
        samples = resample_linear(samples, sr, SR_TARGET)
        sr = SR_TARGET

    total_len = len(samples)
    if total_len < int(0.6 * sr):
        return jsonify({"error": "clip too short - upload/record at least 1-2 seconds"}), 400

    enrolled = np.array(STATE["enrolled_embedding"]) if STATE["enrolled_embedding"] is not None else None

    windows = []
    for idx, (start_sec, end_sec, chunk) in enumerate(chunk_windows(samples, sr)):
        fv, emb, extras = extract_feature_vector(chunk, sr)
        synthetic_prob = DETECTOR.predict_proba(fv)
        similarity = cosine_similarity(emb, enrolled) if enrolled is not None else None
        risk = fuse_risk(synthetic_prob, similarity)

        windows.append({
            "index": idx,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "synthetic_prob": round(synthetic_prob, 4),
            "speaker_similarity": round(similarity, 4) if similarity is not None else None,
            "risk_score": round(risk, 4),
            "risk_label": risk_label(risk),
            "pitch_hz": round(extras["f0_mean"], 1),
            "jitter": round(extras["jitter"], 5),
            "shimmer": round(extras["shimmer"], 5),
        })

    # overall verdict = worst-case window (a single alert window matters)
    order = {"SAFE": 0, "SUSPICIOUS": 1, "ALERT": 2}
    overall = max(windows, key=lambda w: order[w["risk_label"]]) if windows else None

    _, _, full_extras = extract_feature_vector(samples, sr)
    melspec_png = make_melspec_png_b64(full_extras["log_mel"])

    return jsonify({
        "duration_sec": round(total_len / sr, 2),
        "speaker_enrolled": enrolled is not None,
        "windows": windows,
        "overall_risk_label": overall["risk_label"] if overall else "SAFE",
        "overall_risk_score": overall["risk_score"] if overall else 0.0,
        "melspec_png_base64": melspec_png,
    })


@app.route("/api/text-check", methods=["POST"])
def text_check():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = analyze_text(text)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
