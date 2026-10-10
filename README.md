# Voice Clone Detection — Working Prototype

A runnable prototype of the SIH26104 pitch deck: a dual-layer system that
(1) estimates whether audio is AI-generated/synthetic speech and
(2) checks whether the voice matches an enrolled reference speaker,
then fuses both into a Safe / Suspicious / Alert risk decision — analyzed
in short chunks the way a live call would be monitored.

## Quick start

```bash
cd prototype
pip install -r requirements.txt
python backend/app.py
```

Then open **http://localhost:5050** in a browser (Chrome/Edge recommended
for microphone access). Three steps in the UI:

1. **Enroll a Reference Voice** — record or upload 2–5s of a genuine voice.
2. **Analyze a Call** — record or upload the audio to check. You'll get a
   per-window timeline, a detail table (synthetic probability, speaker
   similarity, pitch), and the risk verdict, plus the mel-spectrogram used
   to derive the features.
3. **Check Transcript for AI-Scripted Phrasing** — paste what the caller
   said and get a QuillBot/GPTZero-style "does this read like AI-written
   copy" score, with a breakdown of what drove it (sentence-length
   uniformity, stock phrases, disfluency, etc). See "Text authenticity
   check" below for how this works and its limitations.

No API keys, no cloud services, no GPU required — it runs entirely offline
on CPU.

## What's actually real here

Everything in the pipeline genuinely executes — nothing is hardcoded or
faked for the demo:

- **Feature extraction** (`backend/audio_features.py`): mel-spectrogram,
  MFCCs, pitch/F0 (autocorrelation), jitter, shimmer, spectral flatness,
  centroid, rolloff, and high-frequency energy ratio — all implemented from
  first principles in numpy/scipy (no librosa/torchaudio available in the
  build sandbox — see "Why no PyTorch/librosa" below).
- **Speaker verification**: an MFCC-statistics embedding compared via
  cosine similarity between the enrolled reference and each analysis
  window — the same principle as ECAPA-TDNN/x-vector systems, implemented
  as a lightweight classical version instead of a trained deep embedding.
- **Synthetic-speech classifier** (`backend/model.py`): a real, trained
  scikit-learn MLP (small neural network) that runs inference on every
  window and produces an actual probability, not a random number.
- **Risk fusion + chunked/streaming-style analysis**: implemented exactly
  as described in the pitch deck (`backend/app.py`).

## The one honest limitation: how the classifier was trained

This sandbox has **no internet access**, so it was not possible to download
a labeled deepfake-speech corpus (ASVspoof, WaveFake) or pretrained
AASIST/RawNet2/ECAPA-TDNN weights. To still ship a *genuinely trained,
functioning* classifier rather than a stub, `backend/model.py` bootstraps
its own training data:

1. Synthesizes several "clean voice-like" signals (harmonic synthesis with
   natural pitch jitter, shaped by formant-like resonant filters).
2. Degrades each with `backend/augment.py`, which applies signal-processing
   artifacts known to correlate with vocoder/TTS output (frequency-smoothed
   magnitude spectrum, Griffin-Lim phase re-estimation, bit-depth
   quantization, envelope flattening).
3. Trains an MLP to separate real vs. degraded using the same feature
   pipeline used at inference time.

This is a **proxy task**, not real deepfake detection accuracy — it proves
the full pipeline (features → model → fusion → API → UI) works end to end,
which is what a hackathon prototype needs to demonstrate. Before any real
deployment:

- Retrain `model.py` on ASVspoof 2019/2021 and/or WaveFake with actual
  TTS/voice-conversion samples.
- Consider replacing the MLP with a CNN/AASIST/RawNet2 operating directly
  on the mel-spectrogram (PyTorch), and the speaker module with a trained
  ECAPA-TDNN embedding — the codebase's `extract_feature_vector()` and
  `SyntheticSpeechDetector` are structured so either can be swapped in
  without touching the API or frontend.

## Why no PyTorch/librosa

The build environment had network access disabled, so `pip install
librosa torch soundfile fastapi` etc. could not complete. Everything here
runs on numpy + scipy + scikit-learn + matplotlib + Flask, which were
already available. If you have internet where you run this, feel free to
swap `audio_features.py`'s hand-rolled mel-spectrogram/MFCC for
`librosa.feature.melspectrogram` / `librosa.feature.mfcc` — same output
shape, drop-in replacement.

## Project structure

```
prototype/
├── backend/
│   ├── app.py                Flask API + serves the frontend
│   ├── audio_features.py     Mel-spectrogram, pitch, MFCC, spectral features, WAV parsing
│   ├── augment.py            Vocoder-artifact simulation for bootstrap training data
│   ├── model.py               Bootstrap dataset + MLP classifier train/infer
│   ├── text_authenticity.py  QuillBot/GPTZero-style AI-phrasing heuristic scorer
│   └── retrain.py            Retrain the classifier on real labeled real/fake audio
├── frontend/
│   ├── index.html
│   └── static/
│       ├── styles.css      Same navy/cyan theme as the pitch deck
│       ├── recorder.js     In-browser mic capture -> 16-bit WAV (no libs)
│       └── app.js          UI logic, calls the API, renders results
├── model_store/            Trained classifier gets cached here (.pkl)
└── requirements.txt
```

## Text authenticity check (transcript AI-phrasing detector)

`backend/text_authenticity.py` scores a pasted call transcript for how much
it "reads like AI-generated/scripted text" — the same idea as QuillBot's or
GPTZero's AI-content detectors, applied to what a caller said. This exists
because voice-cloning scams are often scripted with an LLM first, then
either read aloud or spoken by a cloned voice — so scripted phrasing next
to the caller's actual voice is itself a signal worth surfacing.

It scores five things and combines them into a 0–100 "AI-likelihood":
- **Sentence-length uniformity** — spontaneous speech varies sentence
  length a lot; polished/scripted text tends to be more uniform.
- **Lexical diversity** — repeated/narrow vocabulary vs. varied word choice.
- **Stock phrase density** — hits against a small list of stock AI/scam
  transition phrases ("furthermore", "it is important to note", "verify
  your identity", "failure to comply", etc).
- **Lack of disfluency** — natural speech has "um", "like", "you know",
  false starts; scripted text usually doesn't.
- **Repeated sentence openers** — AI text often reuses the same
  transitional openers sentence to sentence.

**Honesty note:** real AI-text detectors (GPTZero, QuillBot, Originality.ai)
run the text through a language model and measure *perplexity* and
*burstiness* — how statistically "surprised" the model is by the text. That
needs a trained LM, and this sandbox has no internet access to download one
(e.g. GPT-2 weights). So this module uses transparent, rule-based
stylistic heuristics instead, which is a reasonable proxy for the same
signal but is not a trained perplexity model. The weighting is hand-tuned
on a handful of test examples, not learned from data — treat the score as
indicative, not authoritative, and swap in a real LM-based perplexity
scorer for production. There's also no offline speech-to-text in this
build (would need Whisper/torch, unavailable without internet), so the
transcript has to be pasted in manually rather than auto-extracted from
the call audio — wiring up STT is the natural next step.

## Retraining on real voice data

`backend/retrain.py` lets you replace the self-supervised bootstrap model
with one trained on actual labeled audio, once you have some:

```bash
cd backend
python retrain.py --real path/to/real_voices --fake path/to/fake_voices
```

- Point `--real` at a folder of genuine human speech `.wav` files (your own
  voice, colleagues, any labeled human speech clips).
- Point `--fake` at a folder of actual AI-generated/cloned `.wav` files
  (output from ElevenLabs, RVC, Tortoise-TTS, or public sets like
  ASVspoof/WaveFake once you can download them).
- Each file is split into the same overlapping windows used at inference
  time, so a handful of clips yields a reasonable number of training
  examples. It holds out a stratified test split by default and prints a
  precision/recall/confusion-matrix report so you can see real accuracy
  (not just the self-consistency smoke test the bootstrap script prints).
- Saves over `model_store/synthetic_detector.pkl` — restart `app.py` (or
  it loads fresh on next launch) to serve the retrained model.

This is the difference between "the pipeline works" (what ships by
default) and "the detector is actually accurate" (what real labeled data
gets you) — see the honesty note above.

## Only WAV is supported

The frontend records audio directly via the Web Audio API and encodes it
to 16-bit PCM WAV in-browser (`recorder.js`), so the backend never has to
decode compressed formats (mp3/webm) — there's no ffmpeg dependency. If you
upload your own file instead of recording, make sure it's a `.wav` file.

## Regenerating the bootstrap model

```bash
cd backend
python model.py
```

This regenerates `model_store/synthetic_detector.pkl` from a fresh
self-supervised bootstrap dataset (see "Why the classifier is a proxy"
above) — not from real labeled data. Use `retrain.py` instead (see
"Retraining on real voice data" above) once you have actual real/fake
audio to train on. Delete the `.pkl` file and restart `app.py` to have it
auto-regenerate the bootstrap version on next launch.
