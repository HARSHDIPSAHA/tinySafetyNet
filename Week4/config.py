import os

# =========================================================
# 1️⃣ BASE DIRECTORY (Prevents path issues)
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# 2️⃣ MODEL & FILE PATHS
# =========================================================

MODEL_PATH = os.path.join(BASE_DIR, "women_safety_dscnn_f16.tflite")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.npy")

# SQLite DB will be created automatically
DB_PATH = os.path.join(BASE_DIR, "safety_data.db")

# =========================================================
# 3️⃣ AUDIO CONFIGURATION
# =========================================================

SAMPLE_RATE = 22050
DURATION = 3.0
CHUNK_DURATION = 0.5  # For realtime app

N_MFCC = 40

TARGET_LENGTH = int(SAMPLE_RATE * DURATION)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
EXPECTED_FRAMES = 130  # Required by your trained model

# =========================================================
# 4️⃣ MQTT CONFIG (Same as your original project)
# =========================================================

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "tinyml/anshika/badge"

# =========================================================
# 5️⃣ SYSTEM METADATA (For Spark Analytics)
# =========================================================

MODEL_VERSION = "v1.0"
DEVICE_ID = "device_001"

# =========================================================
# 6️⃣ RISK LOGIC THRESHOLDS
# =========================================================

CONFIDENCE_THRESHOLD = 50  # Used in realtime app

# Emotion → Risk Mapping
EMOTION_RISK_MAP = {
    "fear": ("🚨 DANGER", "D"),
    "angry": ("⚠️ CAUTION", "C"),
}
# All other emotions default to SAFE

# =========================================================
# 7️⃣ VALIDATION (Prevents silent failures)
# =========================================================

def validate_paths():
    errors = []

    if not os.path.exists(MODEL_PATH):
        errors.append(f"Model file not found at {MODEL_PATH}")

    if not os.path.exists(CLASSES_PATH):
        errors.append(f"Classes file not found at {CLASSES_PATH}")

    if errors:
        raise FileNotFoundError("\n".join(errors))
