import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
from pathlib import Path
from scipy.stats import skew, kurtosis
from pydub import AudioSegment
import io
import tempfile

# ===============================
# Constants
# ===============================
TARGET_SR = 16000
FIXED_DUR = 2.5
N_MFCC    = 20

WAV_DIR = Path("processed") / "wav_clean"
WAV_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# Load model artifacts
# ===============================
@st.cache_resource
def load_artifacts():
    artifact = joblib.load("xgb_model.joblib")
    return artifact["model"], artifact["label_encoder"], artifact["top_features"]

model, label_encoder, top_features = load_artifacts()

# ===============================
# Audio preprocessing
# ===============================
def fix_length_center(y: np.ndarray, sr: int, fixed_dur: float) -> np.ndarray:
    L = int(round(fixed_dur * sr))
    if len(y) < L:
        return np.pad(y, (0, L-len(y)))
    start = (len(y) - L) // 2
    return y[start:start+L]

def preprocess_audio(fp: str) -> np.ndarray:
    y, sr0 = librosa.load(fp, sr=None, mono=True)
    if sr0 != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=TARGET_SR)
    y, _ = librosa.effects.trim(y, top_db=25)
    y = y / (np.max(np.abs(y)) + 1e-8)
    y = fix_length_center(y, TARGET_SR, FIXED_DUR)
    return y

# ===============================
# Feature extraction
# ===============================
def robust_stats(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in ["mean","std","skew","kurt"]}
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std":  float(np.std(x)),
        f"{prefix}_skew": float(skew(x)),
        f"{prefix}_kurt": float(kurtosis(x))
    }

def extract_features_from_audio(y: np.ndarray) -> dict:
    y = y.astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-8)

    feats = {}
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)
    for i in range(N_MFCC):
        feats.update(robust_stats(mfcc[i], f"mfcc{i}"))
        feats.update(robust_stats(d1[i],   f"mfcc{i}_d1"))
        feats.update(robust_stats(d2[i],   f"mfcc{i}_d2"))
    return feats

# ===============================
# Streamlit UI
# ===============================
st.title("🗣️ Speech Disorder Classification")
uploaded_file = st.file_uploader("Upload audio file", type=None)
patient_name = st.text_input("Enter patient name:")

if uploaded_file and patient_name:
    # تحويل أي صوت لـ WAV
    audio_bytes = io.BytesIO(uploaded_file.read())
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(audio_bytes.read())
        tmp.flush()
        audio_seg = AudioSegment.from_file(tmp.name)
        out_wav = WAV_DIR / f"{patient_name}_{Path(uploaded_file.name).stem}.wav"
        audio_seg.export(out_wav, format="wav")

    # Preprocessing + feature extraction
    y = preprocess_audio(str(out_wav))
    features = extract_features_from_audio(y)

    # Align features with model
    X = pd.DataFrame([features])
    for col in top_features:
        if col not in X.columns:
            X[col] = 0.0
    X = X[top_features]

    # Prediction
    probas = model.predict_proba(X)[0]
    pred_idx = np.argmax(probas)
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    # Display result
    st.markdown(f"## Prediction for **{patient_name}**")
    st.write(f"**Predicted label:** {pred_label}")
    st.write("**Probabilities:**")
    for label, prob in zip(label_encoder.classes_, probas):
        st.write(f"{label}: {prob*100:.1f}%")
