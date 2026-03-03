import sqlite3
from contextlib import closing
from config import DB_PATH


# =========================================================
# 1️⃣ DATABASE INITIALIZATION
# =========================================================

def init_db():
    """
    Creates the safety_logs table if it does not exist.
    Safe to call multiple times.
    """

    with sqlite3.connect(DB_PATH) as conn:
        with closing(conn.cursor()) as cursor:

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS safety_logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                source TEXT,
                latitude REAL,
                longitude REAL,
                emotion TEXT,
                confidence REAL,
                risk_level TEXT,
                rms_energy REAL,
                zero_crossing_rate REAL,
                spectral_centroid REAL,
                spectral_bandwidth REAL,
                mfcc_mean REAL,
                mfcc_std REAL,
                duration REAL,
                silence_ratio REAL,
                model_version TEXT,
                device_id TEXT,
                mqtt_signal TEXT,
                processing_time_ms REAL
            )
            """)

            conn.commit()


# =========================================================
# 2️⃣ INSERT LOG ENTRY
# =========================================================

def insert_log(row_tuple):
    """
    Inserts a single row into safety_logs.
    Expects a tuple of 20 values matching table schema.
    """

    try:
        with sqlite3.connect(DB_PATH) as conn:
            with closing(conn.cursor()) as cursor:

                cursor.execute("""
                    INSERT INTO safety_logs VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, row_tuple)

                conn.commit()

    except sqlite3.Error as e:
        # Prevent app crash — log error safely
        print(f"[DB ERROR] {e}")


# =========================================================
# 3️⃣ OPTIONAL: FETCH ALL DATA (Useful for Debug/Spark Export)
# =========================================================

def fetch_all_logs():
    """
    Returns all rows from safety_logs.
    Useful for debugging or exporting.
    """

    with sqlite3.connect(DB_PATH) as conn:
        with closing(conn.cursor()) as cursor:

            cursor.execute("SELECT * FROM safety_logs")
            rows = cursor.fetchall()

    return rows
