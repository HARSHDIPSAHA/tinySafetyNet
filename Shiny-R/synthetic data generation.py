import pandas as pd
import random
from datetime import datetime, timedelta

# Parameters
start_time = datetime.strptime("09:00:00", "%H:%M:%S")
end_time = datetime.strptime("18:00:00", "%H:%M:%S")
interval_seconds = 10

emotions = ["angry", "sad", "happy", "disgust", "neutral"]

# Generate timestamps
timestamps = []
current_time = start_time
while current_time <= end_time:
    timestamps.append(current_time.strftime("%H:%M:%S"))
    current_time += timedelta(seconds=interval_seconds)

# Create dataset
data = {
    "id": range(1, len(timestamps) + 1),
    "time": timestamps,
    "inference_of_emotion": [random.choice(emotions) for _ in timestamps]
}

df = pd.DataFrame(data)

# Save to Excel
output_file = "synthetic_emotion_inference.xlsx"
df.to_excel(output_file, index=False)

print(f"Dataset generated and saved as: {output_file}")
