# 🛡️ AI-Powered Women Safety Badge (IoT + TinyML)

> **Real-time distress detection system using Deep Learning Audio Analysis and IoT alerts.**

This project is a **Smart Safety Badge** simulation that listens to environmental audio in real-time. It uses a **Deep Learning model (DS-CNN)** to detect specific distress emotions like **Fear** (screaming) or **Anger** (aggression). When a threat is detected, it instantly triggers a visual and audio alarm on a wearable badge (simulated via **Wokwi**) over WiFi using **MQTT**.

---

## 🌟 Features

* **🎙️ Real-Time Audio Monitoring:** Continuously listens via microphone using a rolling buffer mechanism.
* **🧠 TinyML Edge AI:** Runs a lightweight **TensorFlow Lite (DS-CNN)** model optimized for speed.
* **📶 IoT Connectivity:** Wireless communication between the Python backend and hardware via **MQTT**.
* **🚨 Instant Alerts:**
* **🔴 Fear/Scream:** Flashing Red LED + High-Pitch Alarm.
* **⚠️ Angry/Aggression:** Yellow LED + Warning Beep.
* **🟢 Safe Environment:** Steady Green LED.
* **📊 Live Dashboard:** A **Streamlit** web interface to visualize confidence scores and detection status.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **AI/ML:** TensorFlow Lite, Librosa (MFCC Feature Extraction)
* **IoT Protocol:** MQTT (Paho-MQTT, HiveMQ Broker)
* **Hardware Simulation:** Wokwi (ESP32)
* **Dashboard:** Streamlit

---

## 📂 Project Structure

```bash

├── 📁 TINYSAFETYNET/streamlit-int8-app/

│   ├── app2.py                # Main Real-Time Streamlit Application

│   ├── app.py                # Streamlit Application to upload audio and get it processed

│   └──  women_safety_dscnn_f16.tflite  # Trained AI Model (Quantized)



```

---

## 🚀 Part 1: Python Setup (The Brain)

### 1. Prerequisites

Ensure you have Python installed. It is recommended to use a virtual environment.

```bash

# Create and activate virtual environment (Windows)

python -m venv venv

.\venv\Scripts\Activate



# Install Dependencies

pip install streamlit numpy tensorflow librosa paho-mqtt pyaudio



```

### 2. Running the System

Run the Streamlit application from your terminal:

```bash

streamlit run app2.py



```

---

## 📟 Part 2: Wokwi Setup (The Badge)

Since we don't have a physical badge, we simulate it using **Wokwi**.

### 1. Create the Project

1. Go to [Wokwi.com](https://wokwi.com).
2. Select **ESP32** (or Arduino, but ESP32 handles WiFi better).
3. **Add Components:** Click the **"+"** button and add:

* 1x **LED (Red)**
* 1x **LED (Yellow)**
* 1x **LED (Green)**
* 1x **Buzzer**
* 3x **Resistors** (220Ω) - *Optional in simulation, but good practice.*

### 2. Wiring Guide

Connect the components to the ESP32 pins as follows:

| Component | Pin (ESP32) | Pin (Component) |

| --- | --- | --- |

| **Red LED** | `GPIO 13` | Anode (+) |

| **Yellow LED** | `GPIO 12` | Anode (+) |

| **Green LED** | `GPIO 14` | Anode (+) |

| **Buzzer** | `GPIO 27` | Positive (+) |

| **All Grounds** | `GND` | Cathode (-) |

### 3. The Firmware Code (`sketch.ino`)

Copy and paste this exact code into the **sketch.ino** tab in Wokwi.

```cpp

#include <WiFi.h>
#include <PubSubClient.h>

// --- YOUR PIN CONFIGURATION ---
#define BUZZER_PIN 0  // D0
#define YELLOW_PIN 15  // D1
#define RED_PIN    5 // D2
#define GREEN_PIN  2  // D3

// --- INTERNET SETTINGS ---
const char* ssid = "Wokwi-GUEST";           // Virtual WiFi
const char* password = "";
const char* mqtt_server = "broker.hivemq.com"; 
const char* topic = "tinyml/anshika/badge"; // Unique ID

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
  Serial.begin(115200);
  
  // Set pins to output mode
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}

// --- NETWORK FUNCTIONS ---
void setup_wifi() {
  Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" Connected!");
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    String clientId = "ESP32Client-";
    clientId += String(random(0xffff), HEX);
    if (client.connect(clientId.c_str())) {
      Serial.println("connected");
      client.subscribe(topic);
    } else {
      delay(5000);
    }
  }
}

// --- ACTION LOGIC ---
void callback(char* topic, byte* payload, unsigned int length) {
  char cmd = (char)payload[0];
  Serial.print("Received Command: ");
  Serial.println(cmd);

  if (cmd == 'S') { // SAFE (Green)
    digitalWrite(GREEN_PIN, HIGH);
    digitalWrite(YELLOW_PIN, LOW);
    digitalWrite(RED_PIN, LOW);
    noTone(BUZZER_PIN);
  }
  else if (cmd == 'C') { // CAUTION (Yellow)
    digitalWrite(GREEN_PIN, LOW);
    digitalWrite(RED_PIN, LOW);
    for(int i=0; i<3; i++){
      digitalWrite(YELLOW_PIN, HIGH);
      tone(BUZZER_PIN, 1000);
      delay(150);
      digitalWrite(YELLOW_PIN, LOW);
      noTone(BUZZER_PIN);
      delay(150);
    }
  }
  else if (cmd == 'D') { // DANGER (Red + Beep)
    digitalWrite(GREEN_PIN, LOW);
    digitalWrite(YELLOW_PIN, LOW);
  
    // Flash 3 times
    for(int i=0; i<3; i++){
      digitalWrite(RED_PIN, HIGH);
      tone(BUZZER_PIN, 1000);
      delay(150);
      digitalWrite(RED_PIN, LOW);
      noTone(BUZZER_PIN);
      delay(150);
    }
  }
}


```

---

## 🕹️ How to Use

1. **Start Wokwi:** Click the green "Play" button in the Wokwi simulation. Wait until you see `WiFi connected` in the Serial Monitor.
2. **Start Python App:** Run `streamlit run app2.py`.
3. **Toggle Start:** Flip the switch labeled **"🔴 START LISTENING"** on the webpage.
4. **Test the Badge:**

* 🗣️ **Speak normally:** Badge turns **Green**.
* 😠 **Shout aggressively:** Badge turns **Yellow** and beeps once.
* 😱 **Scream / Cry for help:** Badge flashes **Red** and sounds a triple alarm.

---

## ⚙️ Configuration (Optional)

You can tweak the `CONFIG` dictionary in `app2.py` to change settings:

```python

CONFIG = {

    "sample_rate": 22050,      # Audio Hz (Must match model training)

    "chunk_duration": 0.5,     # Responsiveness (Lower = Faster updates)

    "mqtt_topic": "tinyml/anshika/badge"  # Change this if you have multiple badges

}



```
