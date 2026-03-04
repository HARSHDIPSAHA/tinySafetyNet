import streamlit as st
import numpy as np
import pyaudio
import time
import uuid
import json
import paho.mqtt.client as mqtt
import streamlit.components.v1 as components

from config import (
    SAMPLE_RATE,
    CHUNK_SAMPLES,
    TARGET_LENGTH,
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_TOPIC,
    CONFIDENCE_THRESHOLD
)

from db_manager import init_db
from model_utils import preprocess_live_audio, predict
from data_collector import log_event


# =========================================================
# 1️⃣ INIT DATABASE
# =========================================================

init_db()


# =========================================================
# 2️⃣ MQTT FUNCTION
# =========================================================

def send_signal(client, command):
    try:
        client.publish(MQTT_TOPIC, command)
    except Exception as e:
        print(f"MQTT Error: {e}")


# =========================================================
# 3️⃣ BROWSER LOCATION
# =========================================================

def get_browser_location():
    geo_script = """
    <script>
    navigator.geolocation.getCurrentPosition(
        function(position) {
            const coords = {
                latitude: position.coords.latitude,
                longitude: position.coords.longitude
            };
            const textarea = window.parent.document.querySelector("textarea");
            textarea.value = JSON.stringify(coords);
            textarea.dispatchEvent(new Event("input", { bubbles: true }));
        }
    );
    </script>
    """
    components.html(geo_script, height=0)

    location_data = st.text_area("geo", height=0)

    if location_data:
        try:
            return json.loads(location_data)
        except:
            return None
    return None


# =========================================================
# 4️⃣ STREAMLIT UI
# =========================================================

st.title("🎙️ Real-Time Safety Guard")
st.markdown("Live emotion detection with geo-tagged logging.")

# --- Location ---
location = get_browser_location()

if location:
    lat = location.get("latitude")
    lon = location.get("longitude")
    st.success(f"Location: {lat:.5f}, {lon:.5f}")
else:
    lat, lon = None, None
    st.warning("Allow location permission.")

# --- UI placeholders ---
status_header = st.empty()
emotion_text = st.empty()
confidence_bar = st.empty()

run_live = st.toggle("🔴 START LISTENING", value=False)


# =========================================================
# 5️⃣ REAL-TIME LOOP
# =========================================================

if run_live:

    # --- Setup Audio ---
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SAMPLES)

    # --- Setup MQTT ---
    mqtt_client = mqtt.Client(client_id=f"rt_client_{uuid.uuid4()}")
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()

    # --- Rolling buffer (3 seconds) ---
    audio_buffer = np.zeros(TARGET_LENGTH, dtype=np.float32)

    st.toast("Microphone Active", icon="🎤")

    try:
        while run_live:

            # Read new chunk
            data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
            new_chunk = np.frombuffer(data, dtype=np.float32)

            # Roll buffer
            audio_buffer = np.roll(audio_buffer, -len(new_chunk))
            audio_buffer[-len(new_chunk):] = new_chunk

            # Volume check
            volume = np.sqrt(np.mean(new_chunk ** 2))

            if volume < 0.01:
                status_header.markdown("## 💤 Status: Silence")
                emotion_text.text("Waiting for sound...")
                continue

            # Prepare input
            input_tensor = preprocess_live_audio(audio_buffer)

            # Predict
            result = predict(input_tensor)

            emotion = result["emotion"]
            confidence = result["confidence"]
            risk = result["risk_label"]

            # Risk threshold logic for live system
            if confidence < CONFIDENCE_THRESHOLD:
                risk = "✅ SAFE"
                result["mqtt_signal"] = "S"

            # Send MQTT
            send_signal(mqtt_client, result["mqtt_signal"])

            # Log structured event
            log_event(
                source="live",
                y=audio_buffer,
                sr=SAMPLE_RATE,
                prediction_result=result,
                latitude=lat,
                longitude=lon
            )

            # Update UI
            if "DANGER" in risk:
                color = "red"
            elif "CAUTION" in risk:
                color = "orange"
            else:
                color = "green"

            status_header.markdown(f"## Status: :{color}[{risk}]")
            emotion_text.markdown(
                f"**Detected:** {emotion.upper()} ({confidence:.1f}%)"
            )

            confidence_bar.progress(int(confidence))

            time.sleep(0.05)

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

        mqtt_client.loop_stop()
        mqtt_client.disconnect()

        st.info("Stopped Listening.")
