import numpy as np
import librosa
import uuid
from datetime import datetime

from config import (
    MODEL_VERSION,
    DEVICE_ID
)

from db_manager import insert_log


# =========================================================
# 1️⃣ FEATURE EXTRACTION
# =========================================================

def extract_audio_features(y, sr):
    """
    Extract additional analytics features for Spark.
    """

    # RMS Energy
    rms_energy = float(np.mean(librosa.feature.rms(y=y)))

    # Zero Crossing Rate
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # Spectral Features
    spectral_centroid = float(np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    ))

    spectral_bandwidth = float(np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sr)
    ))

    # MFCC statistics (not model MFCC — full audio MFCC)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = float(np.mean(mfcc))
    mfcc_std = float(np.std(mfcc))

    # Duration
    duration = float(len(y) / sr)

    # Silence ratio
    silence_ratio = float(np.sum(np.abs(y) < 0.01) / len(y))

    return {
        "rms_energy": rms_energy,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        "duration": duration,
        "silence_ratio": silence_ratio
    }


# =========================================================
# 2️⃣ CREATE FULL EVENT + SAVE TO DATABASE
# =========================================================

def log_event(
    source,
    y,
    sr,
    prediction_result,
    latitude=None,
    longitude=None
):
    """
    Logs a complete structured event to SQLite.

    prediction_result must be dictionary returned from model_utils.predict()
    """

    try:
        features = extract_audio_features(y, sr)

        row = (
            str(uuid.uuid4()),                              # id
            datetime.utcnow().isoformat(),                 # timestamp (UTC safer for Spark)
            source,                                        # upload / live
            latitude,                                      # latitude
            longitude,                                     # longitude
            prediction_result["emotion"],                  # emotion
            prediction_result["confidence"],               # confidence
            prediction_result["risk_label"],               # risk level
            features["rms_energy"],                        # rms
            features["zero_crossing_rate"],                # zcr
            features["spectral_centroid"],                 # centroid
            features["spectral_bandwidth"],                # bandwidth
            features["mfcc_mean"],                         # mfcc mean
            features["mfcc_std"],                          # mfcc std
            features["duration"],                          # duration
            features["silence_ratio"],                     # silence ratio
            MODEL_VERSION,                                 # model version
            DEVICE_ID,                                     # device id
            prediction_result["mqtt_signal"],              # mqtt signal
            prediction_result["processing_time_ms"]        # inference time
        )

        insert_log(row)

    except Exception as e:
        # Do not crash Streamlit
        print(f"[DATA COLLECTOR ERROR] {e}")
