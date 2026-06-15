# 🎙️ Deepfake Audio Detection
Streamlit App: https://527chandaneond-cell-deepfake-audiodetection-app-7pbmig.streamlit.app/

A deep learning system that classifies speech recordings as **Genuine (Human)** or **Deepfake (AI-Generated)**, built on a CNN trained over log-mel spectrograms — with a Streamlit web app for interactive inference.

> **MARS Open Projects 2026** — AI/ML Track, Problem Statement 2: *Deepfake Audio Detection*

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
  - [1. Preprocessing & Feature Extraction](#1-preprocessing--feature-extraction)
  - [2. Model Architecture](#2-model-architecture)
  - [3. Training Setup](#3-training-setup)
- [Results](#-results)
- [Web App](#-web-app)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
  - [Run the Streamlit App](#run-the-streamlit-app)
  - [Run Inference via Script](#run-inference-via-script)
  - [Reproduce Training](#reproduce-training)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)

---

## 🔍 Overview

Advances in generative AI have made it possible to create highly realistic synthetic speech ("deepfake audio"), which can be misused for impersonation, fraud, and social engineering. This project builds an end-to-end pipeline that:

1. Converts raw audio into **log-mel spectrograms**.
2. Feeds them through a **2D Convolutional Neural Network (CNN)**.
3. Outputs a binary classification — **Genuine** vs **Deepfake** — with a confidence score.
4. Serves the model through an interactive **Streamlit** web application.

---

## 🗂 Project Structure

```
.
├── app.py                     # Streamlit web app for inference
├── predict.py                 # Standalone inference script (CLI)
├── notebook.ipynb              # Full reproducible pipeline (EDA → training → evaluation)
├── deepfake_audio_cnn.keras   # Trained CNN model
├── norm_stats.npy             # Saved [mean, std] used for spectrogram normalization
├── performance_report.json    # Evaluation metrics on the held-out test set
├── requirements.txt           # Pinned dependencies
└── README.md                  # Project documentation (this file)
```

---

## 📊 Dataset

**Recommended Dataset:** [The Fake-or-Real (FoR) Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

- The model is trained on the **`for-norm`** (normalized) split of the dataset, using the `training/real` and `training/fake` folders.
- Labels: `0 = Genuine (Human)`, `1 = Deepfake (AI-Generated)`.
- Data is split into **train / validation / test** sets (70% / 15% / 15%) using a **stratified split** to preserve class balance.

---

## 🧪 Methodology

### 1. Preprocessing & Feature Extraction

Each audio clip is converted into a fixed-size **log-mel spectrogram** before being fed to the CNN:

| Step | Detail |
|---|---|
| Resampling | All audio resampled to **16 kHz**, mono |
| Fixed length | Clips padded/trimmed to **4 seconds** (64,000 samples) |
| Spectrogram | Log-Mel Spectrogram via `librosa.feature.melspectrogram` |
| Mel bins | `n_mels = 128` |
| FFT window | `n_fft = 1024` |
| Hop length | `hop_length = 512` |
| Output shape | `128 × 126` (Mel bins × time frames) |
| Normalization | Z-score normalization using **training-set mean & std** (saved in `norm_stats.npy`) |
| Model input | `(128, 126, 1)` — channel dimension added for the CNN |

```python
def audio_to_logmel(y, sr=16000, n_mels=128, n_fft=1024, hop_length=512):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=n_fft, hop_length=hop_length)
    return librosa.power_to_db(mel, ref=np.max)
```

### 2. Model Architecture

A compact **3-block 2D CNN** designed for spectrogram-based audio classification:

```
Input (128 × 126 × 1)
        │
   Conv2D(32, 3×3) → BatchNorm → MaxPool(2×2) → Dropout(0.2)
        │
   Conv2D(64, 3×3) → BatchNorm → MaxPool(2×2) → Dropout(0.2)
        │
   Conv2D(128, 3×3) → BatchNorm → MaxPool(2×2) → Dropout(0.3)
        │
   GlobalAveragePooling2D
        │
   Dense(128, ReLU) → Dropout(0.4)
        │
   Dense(1, Sigmoid)  →  P(Deepfake)
```

- **Output:** A single sigmoid unit producing `P(Deepfake) ∈ [0, 1]`.
- **Decision rule:** `prob ≥ 0.5 → Deepfake`, else `Genuine`.
- **Total parameters:** ~110K (lightweight, fast inference on CPU).

### 3. Training Setup

| Setting | Value |
|---|---|
| Loss function | Binary Cross-Entropy |
| Class imbalance handling | `class_weight` (computed via `sklearn.utils.class_weight.compute_class_weight`) |
| Epochs | Up to 40 (with early stopping) |
| Batch size | 32 |
| Early stopping | Monitors `val_auc`, patience = 8, restores best weights |
| LR scheduling | `ReduceLROnPlateau` on `val_loss` (factor 0.5, patience 4) |
| Checkpointing | Best model saved on `val_auc` |
| Reproducibility | `SEED = 42` across NumPy, Python `random`, and TensorFlow |

---

## 🏆 Results

Evaluated on the held-out **test set**:

| Metric | Score | Required Threshold |
|---|---|---|
| **Overall Accuracy** | **99.96%** | ≥ 80% ✅ |
| **F1 Score** | **99.96%** | ≥ 80% ✅ |
| **Equal Error Rate (EER)** | **0.00%** | ≤ 12% ✅ |
| **Per-Class Accuracy — Genuine** | **99.93%** | ≥ 75% ✅ |
| **Per-Class Accuracy — Deepfake** | **100.00%** | ≥ 75% ✅ |

#### Confusion Matrix

| | Predicted: Genuine | Predicted: Deepfake |
|---|---|---|
| **Actual: Genuine** | 4039 | 3 |
| **Actual: Deepfake** | 0 | 4039 |

> Full machine-readable results are available in [`performance_report.json`](performance_report.json), including the exact preprocessing config used (`sample_rate`, `n_mels`, `n_fft`, `hop_length`, normalization stats).

---

## 🖥 Web App

The Streamlit app (`app.py`) provides an interactive interface to:

- 📤 **Upload an audio file** (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`)
- ▶️ **Play it back** directly in the browser
- 🧠 Get a **Genuine / Deepfake** prediction with a **confidence score**
- 📈 Visualize the **log-mel spectrogram** used as the model input
- 📊 View the model's **performance report** (accuracy, F1, EER, confusion matrix)

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+
- `pip`

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Recommended) Create a virtual environment
conda create -n deepfake_env python=3.10 -y
conda activate deepfake_env
       # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> Ensure the following files are present in the project root before running the app:
> - `deepfake_audio_cnn.keras`
> - `norm_stats.npy`
> - `performance_report.json`

---

## 🚀 Usage

### Run the Streamlit App

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`), upload an audio clip, and view the prediction.

### Run Inference via Script

```bash
python predict.py --input path/to/audio.wav
```

This loads the trained model and normalization stats, preprocesses the audio into a log-mel spectrogram, and prints:

```
Prediction : Deepfake (AI-Generated)
Confidence : 98.42%
```

### Reproduce Training

Open `notebook.ipynb` to walk through the full pipeline:

1. Dataset loading & exploration
2. Feature extraction (log-mel spectrograms)
3. Train / validation / test split + normalization
4. CNN model definition & training
5. Evaluation (accuracy, F1, EER, confusion matrix)
6. Model + report export (`deepfake_audio_cnn.keras`, `performance_report.json`, `norm_stats.npy`)

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Audio Processing | Librosa, SoundFile |
| Data Handling | NumPy, Pandas, scikit-learn |
| Web App | Streamlit |
| Visualization | Matplotlib |

---

## 🔮 Future Improvements

- Cross-dataset evaluation on **ASVspoof 2019** for generalization testing
- Data augmentation (noise injection, pitch/time shifting) for robustness against unseen conditions
- Support for variable-length audio via attention/pooling over chunked spectrograms
- Model explainability (e.g., Grad-CAM over spectrograms to highlight discriminative regions)
- Batch CSV inference mode in the Streamlit app

---

## 📄 License

This project is released for educational purposes as part of **MARS Open Projects 2026**.
