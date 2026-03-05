import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

TOTAL_ROWS = 1_000_000
HOTSPOT_ROWS = 50_000
CHUNK_SIZE = 50000

MODEL_VERSION = "v1.0"
DEVICE_ID = "device_001"

DELHI_CENTER = (28.6139,77.2090)

HOTSPOTS = [
("Paharganj",28.6420,77.2167),
("Karol Bagh",28.6519,77.1909),
("Munirka",28.5562,77.1710),
("Seelampur",28.6691,77.2695),
("Uttam Nagar",28.6219,77.0563),
("Tilak Nagar",28.6365,77.0960),
("Sadar Bazaar",28.6615,77.2167),
("Chandni Chowk",28.6505,77.2303),
("Anand Vihar",28.6469,77.3150),
("Kashmere Gate",28.6675,77.2273),
("Lajpat Nagar",28.5677,77.2436),
("Nehru Place",28.5494,77.2511),
("Kalkaji",28.5490,77.2588),
("Okhla",28.5355,77.2756),
("Jamia Nagar",28.5623,77.2826),
("Rajouri Garden",28.6425,77.1160),
("Punjabi Bagh",28.6682,77.1250),
("Janakpuri",28.6210,77.0810),
("Shahdara",28.6736,77.2890),
("Trilokpuri",28.6132,77.3050),
("Rohini",28.7330,77.1050),
("Mangolpuri",28.6892,77.0820),
("Sultanpuri",28.7031,77.0614),
("Bawana",28.8042,77.0386),
("Narela",28.8527,77.0929),
("Dwarka",28.5562,77.0637),
("Najafgarh",28.6094,76.9855),
("Dhaula Kuan",28.5923,77.1610),
("Noida Sector 15",28.5833,77.3153),
("Noida Sector 18",28.5700,77.3200),
("Noida Sector 62",28.6270,77.3630),
("Noida Sector 137",28.5150,77.4070),
("MG Road Gurgaon",28.4810,77.0900),
("Sikanderpur",28.4820,77.0930),
("IFFCO Chowk",28.4720,77.0720),
("Cyber City",28.4950,77.0880),
("Ghaziabad",28.6692,77.4538),
("Vaishali",28.6465,77.3406),
("Indirapuram",28.6380,77.3700),
("Faridabad",28.4089,77.3178)
]

while len(HOTSPOTS) < 110:
    HOTSPOTS.extend(HOTSPOTS)

HOTSPOTS = HOTSPOTS[:110]


def generate_point(center,scale):
    return (
        np.random.normal(center[0],scale),
        np.random.normal(center[1],scale)
    )


def random_timestamp():
    base = datetime(2026,3,1)
    day = random.randint(0,30)
    hour = random.randint(0,23)
    minute = random.randint(0,59)
    second = random.randint(0,59)

    return (base+timedelta(days=day)).replace(
        hour=hour,
        minute=minute,
        second=second
    ).isoformat()


def risk_assignment(unsafe,danger):

    r=random.random()

    if r<danger:
        return "🚨 DANGER","D"
    elif r<danger+unsafe:
        return "⚠️ CAUTION","C"
    else:
        return "✅ SAFE","S"


def generate_features(risk):

    base_energy=np.random.normal(0.02,0.01)

    if "DANGER" in risk:
        base_energy+=0.03

    return {
        "rms_energy":abs(base_energy),
        "zero_crossing_rate":abs(np.random.normal(0.08,0.02)),
        "spectral_centroid":abs(np.random.normal(1700,300)),
        "spectral_bandwidth":abs(np.random.normal(1750,200)),
        "mfcc_mean":np.random.normal(-7,1),
        "mfcc_std":abs(np.random.normal(70,10)),
        "duration":3.0,
        "silence_ratio":np.clip(np.random.normal(0.6,0.15),0,1),
        "processing_time_ms":abs(np.random.normal(2.0,1.0))
    }


output_file="women_safety_1M_dataset.csv"

generated=0
first_write=True

while generated<TOTAL_ROWS:

    rows=[]

    for i in range(CHUNK_SIZE):

        if generated<HOTSPOT_ROWS:

            name,lat,lon=random.choice(HOTSPOTS)
            lat,lon=generate_point((lat,lon),0.002)

            risk,signal=risk_assignment(.35,.30)

        else:

            lat,lon=generate_point(DELHI_CENTER,0.02)
            risk,signal=risk_assignment(.10,.02)


        emotion=random.choice(["fear","angry","sad","disgust","neutral"])
        confidence=np.clip(np.random.normal(75,15),40,100)

        features=generate_features(risk)

        rows.append({

            "id":str(uuid.uuid4()),
            "timestamp":random_timestamp(),
            "source":"synthetic",
            "latitude":lat,
            "longitude":lon,
            "emotion":emotion,
            "confidence":confidence,
            "risk_level":risk,

            "rms_energy":features["rms_energy"],
            "zero_crossing_rate":features["zero_crossing_rate"],
            "spectral_centroid":features["spectral_centroid"],
            "spectral_bandwidth":features["spectral_bandwidth"],
            "mfcc_mean":features["mfcc_mean"],
            "mfcc_std":features["mfcc_std"],
            "duration":features["duration"],
            "silence_ratio":features["silence_ratio"],

            "model_version":MODEL_VERSION,
            "device_id":DEVICE_ID,
            "mqtt_signal":signal,
            "processing_time_ms":features["processing_time_ms"]
        })

        generated+=1

        if generated>=TOTAL_ROWS:
            break


    df=pd.DataFrame(rows)

    df.to_csv(
        output_file,
        mode="a",
        header=first_write,
        index=False
    )

    first_write=False

    print("generated:",generated)


print("dataset created:",output_file)
