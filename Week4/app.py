import streamlit as st
import tempfile
import os
import uuid
import paho.mqtt.client as mqtt

from streamlit_geolocation import streamlit_geolocation

from config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC
from db_manager import init_db
from model_utils import preprocess_audio_file, predict
from data_collector import log_event


# =========================================================
# INIT DATABASE
# =========================================================
init_db()


# =========================================================
# MQTT FUNCTION
# =========================================================
def send_to_wokwi(command):
    try:
        client_id = f"streamlit_client_{uuid.uuid4()}"
        client = mqtt.Client(client_id=client_id)

        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()

        msg_info = client.publish(MQTT_TOPIC, command, qos=1)
        msg_info.wait_for_publish(timeout=2.0)

        client.loop_stop()
        client.disconnect()
        return True

    except Exception as e:
        st.error(f"MQTT Error: {e}")
        return False


# =========================================================
# UI HEADER
# =========================================================
st.title("🛡️ Women Safety Analytics (Wokwi Linked)")
st.markdown(f"**Connected to:** `{MQTT_TOPIC}` on `{MQTT_BROKER}`")


# =========================================================
# GEOLOCATION (Stable Component)
# =========================================================
location = streamlit_geolocation()

if location and location.get("latitude") and location.get("longitude"):
    lat = location["latitude"]
    lon = location["longitude"]
    st.success(f"📍 Location: {lat:.5f}, {lon:.5f}")
else:
    lat, lon = None, None
    st.info("Allow location access for geo-tagging (optional).")


# =========================================================
# ANALYZE FUNCTION
# =========================================================
def analyze_audio(source):

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        source.seek(0)
        tmp.write(source.read())
        audio_path = tmp.name

    st.audio(audio_path)

    if st.button("Analyze Audio"):

        with st.spinner("Processing..."):

            # Preprocess
            input_tensor, y, sr = preprocess_audio_file(audio_path)

            # Predict
            result = predict(input_tensor)

            emotion = result["emotion"]
            confidence = result["confidence"]
            risk = result["risk_label"]
            mqtt_cmd = result["mqtt_signal"]

            # Send MQTT
            sent = send_to_wokwi(mqtt_cmd)

            # Log event (with location)
            log_event(
                source="upload_or_record",
                y=y,
                sr=sr,
                prediction_result=result,
                latitude=lat,
                longitude=lon
            )

            # Display result
            if "DANGER" in risk:
                color = "red"
            elif "CAUTION" in risk:
                color = "orange"
            else:
                color = "green"

            st.markdown(f"## Result: :{color}[{risk}]")
            st.markdown(f"**Detected Emotion:** {emotion.upper()}")
            st.markdown(f"**Confidence:** {confidence:.2f}%")

            if sent:
                st.success(f"📡 Signal '{mqtt_cmd}' sent to Wokwi Badge.")
            else:
                st.error("❌ Failed to send signal to Wokwi.")

            st.bar_chart(result["probs"])

    # Cleanup
    try:
        os.remove(audio_path)
    except:
        pass


# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs(["📂 Upload File", "🎤 Record Audio"])

with tab1:
    uploaded_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])
    if uploaded_file:
        analyze_audio(uploaded_file)

with tab2:
    recorded_audio = st.audio_input("Record a voice note")
    if recorded_audio:
        analyze_audio(recorded_audio)
