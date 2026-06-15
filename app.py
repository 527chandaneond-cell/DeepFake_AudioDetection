"""
Deepfake Audio Detection — Streamlit Web App
=============================================
Loads a pre-trained CNN (Mel-Spectrogram based) and classifies an
uploaded audio clip as Genuine (Human) or Deepfake (AI-Generated).

Run with:
    streamlit run app.py
"""

import json
import os
import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf

# --------------------------------------------------------------------------- #
# Configuration (must match training config)
# --------------------------------------------------------------------------- #
MODEL_PATH        = "deepfake_audio_cnn.keras"
NORM_STATS_PATH   = "norm_stats.npy"
REPORT_PATH       = "performance_report.json"

SAMPLE_RATE  = 16000
DURATION_SEC = 4
N_MELS       = 128
N_FFT        = 1024
HOP_LENGTH   = 512

CLASS_NAMES = {0: "Genuine (Human)", 1: "Deepfake (AI-Generated)"}

# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    _orig = tf.keras.layers.Dense.__init__
    def _patched(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        _orig(self, *args, **kwargs)
    tf.keras.layers.Dense.__init__ = _patched
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_resource(show_spinner=False)
def load_norm_stats():
    if os.path.exists(NORM_STATS_PATH):
        arr = np.load(NORM_STATS_PATH)
        return float(arr[0]), float(arr[1])
    return -61.5, 19.6   # fallback from performance_report defaults

@st.cache_data(show_spinner=False)
def load_report():
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as f:
            return json.load(f)
    return None

# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def load_audio(file_obj, sr=SAMPLE_RATE, duration=DURATION_SEC):
    y, _ = librosa.load(file_obj, sr=sr, mono=True)
    target = sr * duration
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    else:
        y = y[:target]
    return y

def make_log_mel(y, sr=SAMPLE_RATE, expected_frames=None):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if expected_frames is not None:
        if log_mel.shape[1] < expected_frames:
            log_mel = np.pad(log_mel, ((0,0),(0, expected_frames - log_mel.shape[1])),
                             constant_values=log_mel.min())
        else:
            log_mel = log_mel[:, :expected_frames]
    return log_mel

def preprocess(file_obj, mean, std, expected_frames):
    y       = load_audio(file_obj)
    log_mel = make_log_mel(y, expected_frames=expected_frames)
    normed  = (log_mel - mean) / (std + 1e-9)
    x       = normed[np.newaxis, ..., np.newaxis].astype(np.float32)
    return x, log_mel, y

# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(
        page_title="Deepfake Audio Detection",
        page_icon="🎙️",
        layout="centered",
    )

    st.title("🎙️ Deepfake Audio Detection")
    st.write(
        "Upload an audio clip and the model will classify it as "
        "**Genuine (Human)** or **Deepfake (AI-Generated)**."
    )

    # Load artifacts
    model      = load_model()
    mean, std  = load_norm_stats()
    report     = load_report()

    # ── Dynamically read expected input shape from the model ──────────────── #
    # model.input_shape → (None, N_MELS, T, 1)
    model_input_shape = model.input_shape          # e.g. (None, 128, 125, 1)
    expected_frames   = model_input_shape[2]       # the T dimension
    expected_mels     = model_input_shape[1]

    # ── Sidebar: diagnostics & threshold ─────────────────────────────────── #
    with st.sidebar:
        st.header("⚙️ Settings & Diagnostics")

        threshold = st.slider(
            "Decision threshold (P[Deepfake])",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help=(
                "Lower this if real deepfakes are being predicted as Genuine. "
                "Raise it to be more conservative."
            )
        )

        st.divider()
        st.write("**Model input shape**")
        st.code(str(model_input_shape))
        st.write(f"Expected mel bands : `{expected_mels}`")
        st.write(f"Expected time frames: `{expected_frames}`")
        st.write(f"Norm mean : `{mean:.4f}`")
        st.write(f"Norm std  : `{std:.4f}`")

        if report:
            st.divider()
            st.write("**Training performance**")
            st.metric("Accuracy", f"{report['overall_accuracy']*100:.2f}%")
            st.metric("F1 Score", f"{report['f1_score']*100:.2f}%")
            st.metric("EER",      f"{report['eer']*100:.2f}%")
            st.write("Per-class accuracy:")
            st.json(report["per_class_accuracy"])

    # ── File uploader ─────────────────────────────────────────────────────── #
    uploaded = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
    )

    if uploaded is None:
        st.info("Awaiting audio file upload…")
        return

    st.audio(uploaded)

    with st.spinner("Analysing audio…"):
        uploaded.seek(0)
        x, log_mel, y = preprocess(uploaded, mean, std, expected_frames)
        prob_deepfake  = float(model.predict(x, verbose=0)[0][0])

    # ── Raw output debug box (always shown so you can tune threshold) ─────── #
    with st.expander("🔍 Debug — raw model output", expanded=True):
        st.write(f"**P(Deepfake) = `{prob_deepfake:.4f}`**")
        st.write(f"Decision threshold = `{threshold:.2f}`")
        st.write(f"Spectrogram shape fed to model: `{x.shape}`")
        st.progress(float(prob_deepfake))
        st.caption(
            "If real deepfake audio consistently gives a LOW P(Deepfake), "
            "your training labels were likely flipped. Retrain with labels swapped "
            "OR lower the threshold until calibrated."
        )

    # ── Prediction result ─────────────────────────────────────────────────── #
    label_idx  = 1 if prob_deepfake >= threshold else 0
    label      = CLASS_NAMES[label_idx]
    confidence = prob_deepfake if label_idx == 1 else 1 - prob_deepfake

    st.subheader("Prediction")
    col1, col2 = st.columns(2)
    with col1:
        if label_idx == 1:
            st.error(f"🛑 **{label}**")
        else:
            st.success(f"✅ **{label}**")
    with col2:
        st.metric("Confidence", f"{confidence*100:.2f}%")

    # ── Spectrogram visualisation ─────────────────────────────────────────── #
    with st.expander("View Log-Mel Spectrogram"):
        fig, ax = plt.subplots(figsize=(9, 3))
        img = librosa.display.specshow(
            log_mel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
            x_axis="time", y_axis="mel", ax=ax
        )
        ax.set_title("Log-Mel Spectrogram (CNN input)")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)


if __name__ == "__main__":
    main()