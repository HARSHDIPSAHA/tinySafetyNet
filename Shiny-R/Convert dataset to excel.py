import os
import pandas as pd
from datetime import datetime, timedelta

DATASET_PATH = r"C:\Users\Vibhav\Desktop\Shiny-R\archive"

rows = []

# Synthetic start time
current_time = datetime.strptime("09:00:00", "%H:%M:%S")

# Time gap between recordings (seconds)
TIME_GAP = 3

audio_id = 1

for root, dirs, files in os.walk(DATASET_PATH):
    dirs.sort()     # ensure consistent folder order
    files.sort()    # ensure consistent file order

    for file in files:
        if file.lower().endswith(".wav"):
            emotion = os.path.basename(root).split("_")[-1]

            rows.append({
                "id": audio_id,
                "time": current_time.strftime("%H:%M:%S"),
                "inference_of_emotion": emotion
            })

            current_time += timedelta(seconds=TIME_GAP)
            audio_id += 1

df = pd.DataFrame(rows)
df.to_excel("tess_emotion_log.xlsx", index=False)

print("Excel created: tess_emotion_log.xlsx")
