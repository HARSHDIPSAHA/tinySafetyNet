import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ==========================
# CONFIG
# ==========================
TOTAL_ROWS = 5000
DELHI_CENTER = (28.6139, 77.2090)

SAROJINI_CENTER = (28.5775, 77.1960)
NSUT_CENTER = (28.6090, 77.0380)

MODEL_VERSION = "v1.0"
DEVICE_ID = "device_001"

# ==========================
# HELPER FUNCTIONS
# ==========================

def generate_point(center, scale=0.01):
    return (
        np.random.normal(center[0], scale),
        np.random.normal(center[1], scale)
    )

def random_timestamp(market=False):
    base_date = datetime(2026, 3, 1)
    day_offset = random.randint(0, 5)

    if market:
        # Afternoon & Evening bias
        hour = random.choice(
            list(range(13, 18)) + list(range(17, 22))
        )
    else:
        hour = random.randint(6, 23)

    minute = random.randint(0, 59)
    second = random.randint(0, 59)

    return (base_date + timedelta(days=day_offset)).replace(
        hour=hour, minute=minute, second=second
    ).isoformat()


def risk_assignment(prob_unsafe, prob_danger):
    r = random.random()

    if r < prob_danger:
        return "🚨 DANGER", "D"
    elif r < prob_danger + prob_unsafe:
        return "⚠️ CAUTION", "C"
    else:
        return "✅ SAFE", "S"


def generate_features(risk_level):
    base_energy = np.random.normal(0.02, 0.01)

    if "DANGER" in risk_level:
        base_energy += 0.03

    return {
        "rms_energy": abs(base_energy),
        "zero_crossing_rate": abs(np.random.normal(0.08, 0.02)),
        "spectral_centroid": abs(np.random.normal(1700, 300)),
        "spectral_bandwidth": abs(np.random.normal(1750, 200)),
        "mfcc_mean": np.random.normal(-7, 1),
        "mfcc_std": abs(np.random.normal(70, 10)),
        "duration": 3.0,
        "silence_ratio": np.clip(np.random.normal(0.6, 0.15), 0, 1),
        "processing_time_ms": abs(np.random.normal(2.0, 1.0))
    }


# ==========================
# DATA GENERATION
# ==========================

rows = []

for i in range(TOTAL_ROWS):

    region_selector = random.random()

    # 10% Sarojini Market
    if region_selector < 0.10:
        lat, lon = generate_point(SAROJINI_CENTER, scale=0.003)
        risk, signal = risk_assignment(prob_unsafe=0.28, prob_danger=0.44)  # 72% unsafe/danger
        timestamp = random_timestamp(market=True)

    # 30% NSUT
    elif region_selector < 0.40:
        lat, lon = generate_point(NSUT_CENTER, scale=0.005)
        risk, signal = risk_assignment(prob_unsafe=0.10, prob_danger=0.02)
        timestamp = random_timestamp()

    # 10% Metro Corridor
    elif region_selector < 0.50:
        # Simulate line between two points
        t = random.random()
        lat = 28.50 + t * (28.70 - 28.50)
        lon = 77.00 + t * (77.30 - 77.00)
        risk, signal = risk_assignment(prob_unsafe=0.20, prob_danger=0.03)
        timestamp = random_timestamp()

    # Remaining 50% General Delhi (Normal Distribution)
    else:
        lat, lon = generate_point(DELHI_CENTER, scale=0.02)
        risk, signal = risk_assignment(prob_unsafe=0.08, prob_danger=0.01)
        timestamp = random_timestamp()

    emotion = random.choice(["fear", "angry", "sad", "disgust", "neutral"])
    confidence = np.clip(np.random.normal(75, 15), 40, 100)

    features = generate_features(risk)

    rows.append({
        "id": str(uuid.uuid4()),
        "timestamp": timestamp,
        "source": "synthetic",
        "latitude": lat,
        "longitude": lon,
        "emotion": emotion,
        "confidence": confidence,
        "risk_level": risk,
        "rms_energy": features["rms_energy"],
        "zero_crossing_rate": features["zero_crossing_rate"],
        "spectral_centroid": features["spectral_centroid"],
        "spectral_bandwidth": features["spectral_bandwidth"],
        "mfcc_mean": features["mfcc_mean"],
        "mfcc_std": features["mfcc_std"],
        "duration": features["duration"],
        "silence_ratio": features["silence_ratio"],
        "model_version": MODEL_VERSION,
        "device_id": DEVICE_ID,
        "mqtt_signal": signal,
        "processing_time_ms": features["processing_time_ms"]
    })


df = pd.DataFrame(rows)
df.to_csv("synthetic_safety_dataset.csv", index=False)

print("Synthetic dataset generated successfully.")
print("File saved as: synthetic_safety_dataset.csv")
print("Total rows:", len(df))
