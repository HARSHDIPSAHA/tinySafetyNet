import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import tempfile
import os

# ==========================================
# CONFIG (MATCHES YOUR CODE EXACTLY)
# ==========================================
TARGET_SR = 16000
N_MELS = 64

ID_TO_LABEL = {
    0: "Safe/Neutral",
    1: "DANGER (Fear)",
    2: "Caution (Angry)"
}

# ==========================================
# LOAD TFLITE MODEL
# ==========================================
@st.cache_resource
def load_interpreter():
    interpreter = tf.lite.Interpreter(
        model_path="tiny_safety_3class_int8.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_interpreter()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

EXPECTED_SHAPE = tuple(input_details[0]["shape"])  # (1, 1, 64, 64)

# ==========================================
# PREPROCESS (BIT-EXACT WITH inference.py)
# ==========================================
def preprocess_audio(audio_np):
    # Normalize waveform
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val

    # Mel Spectrogram (LIBROSA — SAME AS TRAINING)
    mel = librosa.feature.melspectrogram(
        y=audio_np,
        sr=TARGET_SR,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize to (64, 64) — MATCH torch.interpolate
    mel_db = tf.image.resize(
        mel_db[:, :, np.newaxis],
        (64, 64)
    ).numpy()[:, :, 0]


    # Normalize 0–1 (MATCH training)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    # NCHW FORMAT (CRITICAL)
    mel_db = mel_db[np.newaxis, np.newaxis, :, :]  # (1, 1, 64, 64)

    # Shape validation (SAFE)
    if mel_db.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Input shape mismatch: got {mel_db.shape}, expected {EXPECTED_SHAPE}"
        )

    # INT8 quantization
    if input_details[0]["dtype"] == np.int8:
        scale, zero_point = input_details[0]["quantization"]
        mel_db = mel_db / scale + zero_point
        mel_db = np.clip(mel_db, -128, 127).astype(np.int8)

    return mel_db

# ==========================================
# STREAMLIT UI
# ==========================================
st.title("🔊 TinyML Safety Audio Classifier (INT8)")
st.write("Upload an audio file to analyze emotional safety state.")

uploaded_file = st.file_uploader(
    "Upload WAV or MP3",
    type=["wav", "mp3"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    try:
        audio, _ = librosa.load(audio_path, sr=TARGET_SR)

        input_tensor = preprocess_audio(audio)

        interpreter.set_tensor(
            input_details[0]["index"],
            input_tensor
        )
        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        # Dequantize output if INT8
        if output_details[0]["dtype"] == np.int8:
            scale, zero_point = output_details[0]["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale

        probs = tf.nn.softmax(output).numpy()

        pred_id = int(np.argmax(probs))
        pred_label = ID_TO_LABEL[pred_id]

        st.success(f"Prediction: **{pred_label}**")

        st.subheader("Confidence Scores")
        for i, label in ID_TO_LABEL.items():
            st.write(f"{label}: {probs[i]*100:.1f}%")

    except Exception as e:
        st.error(f"Error during inference: {e}")

    finally:
        os.remove(audio_path)
