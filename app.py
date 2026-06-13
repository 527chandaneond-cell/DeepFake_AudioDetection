"""
Deepfake Audio Detection — Streamlit Web App
=============================================
Loads a pre-trained CNN (Mel-Spectrogram based) and classifies an
uploaded audio clip as Genuine (Human) or Deepfake (AI-Generated),
returning a confidence score.

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
# Configuration (must match the config used during training)
# --------------------------------------------------------------------------- #
MODEL_PATH = "deepfake_audio_cnn.keras"
NORM_STATS_PATH = "norm_stats.npy"
PERFORMANCE_REPORT_PATH = "performance_report.json"

SAMPLE_RATE = 16000
DURATION_SEC = 4
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
EXPECTED_FRAMES = 126  # time-axis size expected by the CNN input (128, 126, 1)

CLASS_NAMES = {0: "Genuine", 1: "Deepfake"}


# --------------------------------------------------------------------------- #
# Cached resource loaders
# --------------------------------------------------------------------------- #
def _strip_unsupported_keys(config):
    """Recursively remove config keys that newer Keras adds but older
    Keras versions don't recognize (e.g. 'quantization_config'),
    so the model can be deserialized across Keras versions."""
    unsupported_keys = {"quantization_config"}

    if isinstance(config, dict):
        for key in unsupported_keys:
            config.pop(key, None)
        for value in config.values():
            _strip_unsupported_keys(value)
    elif isinstance(config, list):
        for item in config:
            _strip_unsupported_keys(item)

    return config
_original_dense_init = tf.keras.layers.Dense.__init__

def patched_dense_init(self, *args, **kwargs):
    # If Kaggle's 'quantization_config' is in the saved file, throw it in the trash
    kwargs.pop('quantization_config', None)
    _original_dense_init(self, *args, **kwargs)

# Apply the patch to TensorFlow
tf.keras.layers.Dense.__init__ = patched_dense_init
# ==========================================

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    
    # Ensure MODEL_PATH is pointing to your 'deepfake_audio_cnn.keras' file
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

@st.cache_resource(show_spinner=False)
def load_norm_stats():
    if os.path.exists(NORM_STATS_PATH):
        stats = np.load(NORM_STATS_PATH)
        return float(stats[0]), float(stats[1])
    # Fallback to values reported in performance_report.json
    return -61.501808166503906, 19.617700576782227


@st.cache_data(show_spinner=False)
def load_performance_report():
    if os.path.exists(PERFORMANCE_REPORT_PATH):
        with open(PERFORMANCE_REPORT_PATH, "r") as f:
            return json.load(f)
    return None


# --------------------------------------------------------------------------- #
# Audio preprocessing
# --------------------------------------------------------------------------- #
def load_audio(file_obj, sr=SAMPLE_RATE, duration=DURATION_SEC):
    """Load an audio file, resample, and force it to a fixed length."""
    y, _ = librosa.load(file_obj, sr=sr, mono=True)

    target_len = sr * duration
    if len(y) < target_len:
        # Pad with zeros (silence) if the clip is shorter than 4 seconds
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        # Trim to the first 4 seconds
        y = y[:target_len]

    return y


def extract_log_mel_spectrogram(y, sr=SAMPLE_RATE):
    """Compute a log-mel spectrogram and normalize it using training stats."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Ensure a fixed time dimension (pad / trim to EXPECTED_FRAMES)
    if log_mel.shape[1] < EXPECTED_FRAMES:
        pad_width = EXPECTED_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=log_mel.min())
    elif log_mel.shape[1] > EXPECTED_FRAMES:
        log_mel = log_mel[:, :EXPECTED_FRAMES]

    return log_mel


def preprocess_audio(file_obj, mean, std):
    y = load_audio(file_obj)
    log_mel = extract_log_mel_spectrogram(y)

    # Normalize using the dataset mean/std (saved in norm_stats.npy)
    norm_mel = (log_mel - mean) / std

    # Shape -> (1, N_MELS, EXPECTED_FRAMES, 1)
    model_input = norm_mel[np.newaxis, ..., np.newaxis].astype(np.float32)
    return model_input, log_mel, y


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict(model, model_input):
    prob_deepfake = float(model.predict(model_input, verbose=0)[0][0])
    label = 1 if prob_deepfake >= 0.5 else 0
    confidence = prob_deepfake if label == 1 else 1 - prob_deepfake
    return CLASS_NAMES[label], confidence, prob_deepfake


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(
        page_title="Deepfake Audio Detection",
        page_icon="🎙️",
        layout="centered",
    )

    st.title("🎙️ Deepfake Audio Detection")
    st.write(
        "Upload an audio clip and the model will classify it as **Genuine "
        "(Human)** or **Deepfake (AI-Generated)**, along with a confidence score."
    )

    mean, std = load_norm_stats()

    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        with st.spinner("Analyzing audio..."):
            model = load_model()
            model_input, log_mel, y = preprocess_audio(uploaded_file, mean, std)
            label, confidence, prob_deepfake = predict(model, model_input)

        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            if label == "Deepfake":
                st.error(f"🛑 **{label}**")
            else:
                st.success(f"✅ **{label}**")
        with col2:
            st.metric("Confidence", f"{confidence * 100:.2f}%")

        st.progress(prob_deepfake)
        st.caption(
            f"Raw model output (P(Deepfake)) = {prob_deepfake:.4f} "
            f"— threshold = 0.50"
        )

        with st.expander("View Mel-Spectrogram"):
            fig, ax = plt.subplots(figsize=(8, 4))
            librosa.display.specshow(
                log_mel,
                sr=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                x_axis="time",
                y_axis="mel",
                ax=ax,
            )
            ax.set_title("Log-Mel Spectrogram (input to the CNN)")
            fig.colorbar(ax.images[0], ax=ax, format="%+2.0f dB")
            st.pyplot(fig)

    st.divider()

    report = load_performance_report()
    if report:
        with st.expander("Model Performance Report"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall Accuracy", f"{report['overall_accuracy'] * 100:.2f}%")
            c2.metric("F1 Score", f"{report['f1_score'] * 100:.2f}%")
            c3.metric("EER", f"{report['eer'] * 100:.2f}%")

            st.write("**Per-Class Accuracy**")
            st.json(report["per_class_accuracy"])

            st.write("**Confusion Matrix** (rows = true, cols = predicted)")
            st.write(
                "Order: [Genuine, Deepfake]\n\n"
                f"{np.array(report['confusion_matrix'])}"
            )


if __name__ == "__main__":
    main()