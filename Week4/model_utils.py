import numpy as np
import tensorflow as tf
import librosa
import os
import time

from config import (
    MODEL_PATH,
    CLASSES_PATH,
    SAMPLE_RATE,
    TARGET_LENGTH,
    EXPECTED_FRAMES,
    N_MFCC,
    EMOTION_RISK_MAP,
    CONFIDENCE_THRESHOLD,
    validate_paths
)

# =========================================================
# 1️⃣ VALIDATE FILES
# =========================================================

validate_paths()

# =========================================================
# 2️⃣ LOAD MODEL + CLASSES (Singleton Style)
# =========================================================

_interpreter = None
_classes = None


def load_model():
    global _interpreter, _classes

    if _interpreter is None:
        _interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()

    if _classes is None:
        if os.path.exists(CLASSES_PATH):
            _classes = np.load(CLASSES_PATH, allow_pickle=True)
        else:
            _classes = np.array(
                ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            )

    return _interpreter, _classes


# =========================================================
# 3️⃣ PREPROCESS AUDIO (Used by Upload Version)
# =========================================================

def preprocess_audio_file(audio_path):
    """
    Loads audio from file and prepares it for inference.
    Returns:
        input_tensor, raw_audio_array, sample_rate
    """

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Trim silence
    y, _ = librosa.effects.trim(y)

    # Pad / truncate to fixed length
    if len(y) > TARGET_LENGTH:
        y = y[:TARGET_LENGTH]
    else:
        padding = TARGET_LENGTH - len(y)
        y = np.pad(y, (0, padding), 'constant')

    input_tensor = _prepare_model_input(y, sr)

    return input_tensor, y, sr


# =========================================================
# 4️⃣ PREPROCESS LIVE AUDIO (Realtime Version)
# =========================================================

def preprocess_live_audio(audio_buffer):
    """
    Takes rolling buffer from realtime app.
    Returns input tensor only.
    """

    input_tensor = _prepare_model_input(audio_buffer, SAMPLE_RATE)
    return input_tensor


# =========================================================
# 5️⃣ INTERNAL: PREPARE MODEL INPUT
# =========================================================

def _prepare_model_input(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.astype(np.float32)

    # Fix time frames
    current_frames = mfcc.shape[1]

    if current_frames < EXPECTED_FRAMES:
        pad_width = EXPECTED_FRAMES - current_frames
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :EXPECTED_FRAMES]

    # Add batch + channel dims
    input_tensor = np.expand_dims(mfcc, axis=0)
    input_tensor = np.expand_dims(input_tensor, axis=-1)

    return input_tensor


# =========================================================
# 6️⃣ RUN INFERENCE
# =========================================================

def predict(input_tensor):
    """
    Runs inference and returns:
        emotion, confidence, risk_label, mqtt_signal, raw_probs
    """

    interpreter, classes = load_model()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    start_time = time.time()

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]

    processing_time = (time.time() - start_time) * 1000

    pred_idx = np.argmax(probs)
    confidence = float(probs[pred_idx] * 100)
    emotion = str(classes[pred_idx])

    # Determine risk
    if emotion in EMOTION_RISK_MAP and confidence >= CONFIDENCE_THRESHOLD:
        risk_label, mqtt_signal = EMOTION_RISK_MAP[emotion]
    else:
        risk_label = "✅ SAFE"
        mqtt_signal = "S"

    return {
        "emotion": emotion,
        "confidence": confidence,
        "risk_label": risk_label,
        "mqtt_signal": mqtt_signal,
        "probs": probs,
        "processing_time_ms": processing_time
    }
