"""
audio_features.py
------------------
Hand-rolled acoustic feature extraction using only numpy + scipy.

This exists because the sandbox this prototype was built in has no internet
access, so librosa / torchaudio could not be installed. Every feature here
(mel-spectrogram, F0/pitch, jitter, shimmer, spectral flatness/centroid/
rolloff, MFCC) is implemented from first principles. In a networked dev
environment you would normally just `import librosa` -- swap that in for
production; the function signatures below (`extract_feature_vector`,
`mel_spectrogram`) are designed to be a drop-in replacement point.
"""

import io
import wave
import numpy as np
from scipy.fftpack import dct

SR_TARGET = 16000          # we resample everything to 16kHz internally
FRAME_MS = 25
HOP_MS = 10
N_MELS = 40
N_MFCC = 13
F0_MIN, F0_MAX = 70, 400   # human voice fundamental frequency range (Hz)
WINDOW_SEC = 1.5           # analysis chunk size -- shared by app.py and retrain.py
HOP_SEC = 1.0              # chunk stride (overlapping windows)


def chunk_windows(samples: np.ndarray, sr: int, window_sec=WINDOW_SEC, hop_sec=HOP_SEC, min_sec=0.4):
    """Yields (start_sec, end_sec, chunk) for overlapping analysis windows.
    Used identically at inference time (app.py) and training time
    (retrain.py) so a trained model sees the same chunk statistics it will
    be scored on live."""
    win_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)
    total_len = len(samples)
    start = 0
    while start < total_len:
        end = min(start + win_len, total_len)
        chunk = samples[start:end]
        if len(chunk) < int(min_sec * sr):
            break
        yield start / sr, end / sr, chunk
        start += hop_len


# ---------------------------------------------------------------- resampling
def resample_linear(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return samples
    duration = len(samples) / sr_in
    n_out = int(round(duration * sr_out))
    x_old = np.linspace(0, duration, num=len(samples), endpoint=False)
    x_new = np.linspace(0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


# ---------------------------------------------------------------- framing
def frame_signal(samples: np.ndarray, sr: int, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    if len(samples) < frame_len:
        samples = np.pad(samples, (0, frame_len - len(samples)))
    n_frames = 1 + (len(samples) - frame_len) // hop_len
    frames = np.stack([
        samples[i * hop_len: i * hop_len + frame_len] for i in range(max(n_frames, 1))
    ])
    window = np.hanning(frame_len)
    return frames * window, frame_len, hop_len


# ---------------------------------------------------------------- mel filterbank
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels=N_MELS, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        f_center = max(f_center, f_left + 1)
        f_right = max(f_right, f_center + 1)
        for k in range(f_left, f_center):
            if 0 <= k < fbank.shape[1]:
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if 0 <= k < fbank.shape[1]:
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    return fbank


# ---------------------------------------------------------------- mel-spectrogram / MFCC
def mel_spectrogram(samples: np.ndarray, sr: int, n_mels=N_MELS):
    frames, frame_len, hop_len = frame_signal(samples, sr)
    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (np.abs(spec) ** 2) / n_fft
    fbank = mel_filterbank(sr, n_fft, n_mels=n_mels)
    mel_energies = power @ fbank.T                      # (n_frames, n_mels)
    mel_energies = np.maximum(mel_energies, 1e-10)
    log_mel = np.log(mel_energies)
    return log_mel, power, hop_len


def mfcc_from_logmel(log_mel: np.ndarray, n_mfcc=N_MFCC):
    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return coeffs


# ---------------------------------------------------------------- pitch (autocorrelation)
def estimate_pitch_track(samples: np.ndarray, sr: int):
    frames, frame_len, hop_len = frame_signal(samples, sr)
    lag_min = int(sr / F0_MAX)
    lag_max = min(int(sr / F0_MIN), frame_len - 1)
    f0s = []
    for frame in frames:
        frame = frame - frame.mean()
        ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        if ac[0] <= 1e-9:
            f0s.append(0.0)
            continue
        ac_norm = ac / ac[0]
        segment = ac_norm[lag_min:lag_max]
        if len(segment) == 0:
            f0s.append(0.0)
            continue
        peak_idx = int(np.argmax(segment))
        peak_val = segment[peak_idx]
        lag = lag_min + peak_idx
        f0 = sr / lag if peak_val > 0.3 else 0.0   # voicing threshold
        f0s.append(f0)
    return np.array(f0s)


def jitter_shimmer(samples: np.ndarray, sr: int, f0_track: np.ndarray):
    frames, frame_len, hop_len = frame_signal(samples, sr)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

    voiced = f0_track > 0
    voiced_f0 = f0_track[voiced]
    if len(voiced_f0) > 2:
        periods = 1.0 / voiced_f0
        diffs = np.abs(np.diff(periods))
        jitter = float(np.mean(diffs) / (np.mean(periods) + 1e-9))
    else:
        jitter = 0.0

    voiced_rms = rms[voiced] if voiced.sum() > 2 else rms
    if len(voiced_rms) > 2:
        diffs = np.abs(np.diff(voiced_rms))
        shimmer = float(np.mean(diffs) / (np.mean(voiced_rms) + 1e-9))
    else:
        shimmer = 0.0
    return jitter, shimmer, rms


# ---------------------------------------------------------------- spectral stats
def spectral_stats(power: np.ndarray, sr: int, n_fft: int):
    freqs = np.linspace(0, sr / 2, power.shape[1])
    eps = 1e-12

    geo_mean = np.exp(np.mean(np.log(power + eps), axis=1))
    arith_mean = np.mean(power, axis=1) + eps
    flatness = geo_mean / arith_mean

    total_energy = np.sum(power, axis=1) + eps
    centroid = np.sum(power * freqs[None, :], axis=1) / total_energy

    cumulative = np.cumsum(power, axis=1)
    rolloff_thresh = 0.85 * total_energy
    rolloff = np.array([
        freqs[np.searchsorted(cumulative[i], rolloff_thresh[i])]
        if np.searchsorted(cumulative[i], rolloff_thresh[i]) < len(freqs) else freqs[-1]
        for i in range(power.shape[0])
    ])

    hf_mask = freqs > 4000
    hf_ratio = np.sum(power[:, hf_mask], axis=1) / total_energy

    return flatness, centroid, rolloff, hf_ratio


# ---------------------------------------------------------------- top-level extraction
def extract_feature_vector(samples: np.ndarray, sr: int):
    """Returns (feature_vector, embedding_vector, extras dict)."""
    if sr != SR_TARGET:
        samples = resample_linear(samples, sr, SR_TARGET)
        sr = SR_TARGET

    log_mel, power, hop_len = mel_spectrogram(samples, sr)
    mfcc = mfcc_from_logmel(log_mel)
    f0_track = estimate_pitch_track(samples, sr)
    jitter, shimmer, rms = jitter_shimmer(samples, sr, f0_track)
    flatness, centroid, rolloff, hf_ratio = spectral_stats(power, sr, n_fft=power.shape[1] * 2 - 2)

    voiced_f0 = f0_track[f0_track > 0]
    f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) else 0.0
    f0_std = float(np.std(voiced_f0)) if len(voiced_f0) else 0.0

    scalar_feats = np.array([
        f0_mean, f0_std, jitter, shimmer,
        float(np.mean(flatness)), float(np.std(flatness)),
        float(np.mean(centroid)), float(np.std(centroid)),
        float(np.mean(rolloff)), float(np.mean(hf_ratio)),
    ])
    mfcc_mean = mfcc.mean(axis=0)
    mfcc_std = mfcc.std(axis=0)

    feature_vector = np.concatenate([scalar_feats, mfcc_mean, mfcc_std])
    embedding = np.concatenate([mfcc_mean, mfcc_std])  # used for speaker similarity

    extras = {
        "log_mel": log_mel,
        "f0_track": f0_track.tolist(),
        "jitter": jitter,
        "shimmer": shimmer,
        "flatness_mean": float(np.mean(flatness)),
        "centroid_mean": float(np.mean(centroid)),
        "hf_ratio_mean": float(np.mean(hf_ratio)),
        "f0_mean": f0_mean,
    }
    return feature_vector, embedding, extras


def read_wav_bytes(raw: bytes):
    """Parse WAV bytes (16/24/32-bit PCM or 8-bit unsigned, mono or multi-
    channel) into a float32 mono waveform in [-1, 1]. Shared by the Flask
    API (app.py, given file-upload bytes) and retrain.py (given bytes read
    from disk), so both go through identical parsing."""
    with wave.open(io.BytesIO(raw), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, sr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-9)
    b_n = b / (np.linalg.norm(b) + 1e-9)
    return float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
