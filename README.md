# 🛡️ TinySafetyNET

> **Real-time distress detection system using Deep Learning Audio Analysis and IoT alerts.**

This project is a **Smart Safety Badge** simulation that listens to environmental audio in real-time. It uses a **Deep Learning model (DS-CNN)** to detect specific distress emotions like **Fear** (screaming) or **Anger** (aggression). When a threat is detected, it instantly triggers a visual and audio alarm on a wearable badge (simulated via **Wokwi**) over WiFi using **MQTT**.

Interestingly, my model is just 40 KB, small enough to fit even on ultra-low flash memory IoT devices.
This is what real TinyML feels like — efficient, deployable, and impactful.

Using such lightweight intelligence for women safety applications makes it even more meaningful. 🫶---

## 🌟 Features

* **🎙️ Real-Time Audio Monitoring:** Continuously listens via microphone using a rolling buffer mechanism.
* **🧠 TinyML Edge AI:** Runs a lightweight **TensorFlow Lite (DS-CNN)** model optimized for speed.
* **📶 IoT Connectivity:** Wireless communication between the Python backend and hardware via **MQTT**.
* **🚨 Instant Alerts:**
* **🔴 Fear/Scream:** Flashing Red LED + High-Pitch Alarm.
* **⚠️ Angry/Aggression:** Yellow LED + Warning Beep.
* **🟢 Safe Environment:** Steady Green LED.
* **📊 Live Dashboard:** A **Streamlit** web interface to visualize confidence scores and detection status.
* 🔍 **Automated Data Validation Pipelines**  
* 🐳 Fully containerized using **Docker**
* ☸️ Orchestrated via **Kubernetes (Minikube)**
* 🔁 CI/CD enabled through **GitHub Actions**

---

## 📊 Polyglot Dashboards

### 🐍 Live Inference Dashboard (Streamlit)
- Real-time confidence scores
- Inference state visualization
- MQTT status tracking

### 📈 Analytics Dashboard (R Shiny)
- Historical emotion distribution
- Hour-wise heatmaps
- Class frequency charts
- Interactive filters

## 🏭 Enterprise MLOps
- Docker containerization
- Kubernetes deployment
- Automated data validation jobs
- CI/CD workflows

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **AI/ML:** TensorFlow Lite, Librosa (MFCC Feature Extraction)
* **IoT Protocol:** MQTT (Paho-MQTT, HiveMQ Broker)
* **Hardware Simulation:** Wokwi (ESP32)
* **Dashboard:** Streamlit
* **GitHub Actions:** Automated CI/CD pipeline for validation 
* **Kubernetes:**: dockerized containers for both R app and the streamlit dashboard

---

## 🧩 Model Architecture (DS-CNN)

A simple view of the model used for audio emotion detection. It takes MFCC features as input and produces a probability over classes.

```mermaid
flowchart TD
    A["Input: MFCC features (40 x T x 1)"]
    B["Conv layer (Downsampling)"]
    C["Depthwise-Separable Conv Block 1"]
    D["Depthwise-Separable Conv Block 2"]
    E["Global Average Pooling"]
    F["Dropout (rate = 0.4)"]
    G["Dense Layer + Softmax"]
    H["Output: Emotion Class"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H


```

---

# 📂 Project Structure

```bash
tinySafetyNet/
│
├── .github/workflows/          # CI/CD pipelines
│
├── k8s/                        # Kubernetes manifests
│   ├── streamlit-deployment.yaml
│   ├── shiny-deployment.yaml
│   └── data-ops-cronjob.yaml
│
├── week1/                      # Python inference service
│   └── streamlit-int8-app/
│       ├── Dockerfile
│       ├── app.py
│       ├── requirements.txt
│       ├── classes.npy
│       └── women_safety_dscnn_f16.tflite
│
├── week2/                      # R analytics service
│   ├── Dockerfile
│   ├── app.R
│   └── tess_emotion_log.xlsx
│
└── week5_ops/                  # Data validation layer
    └── data_validator.py                       
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

| Component   | Pin (ESP32) | Pin (Component) |
| ---         | ---         | ---             |
| Red LED     | GPIO 13     | Anode (+)       |
| Yellow LED  | GPIO 12     | Anode (+)       |
| Green LED   | GPIO 14     | Anode (+)       |
| Buzzer      | GPIO 27     | Positive (+)    |
| All Grounds | GND         | Cathode (-)     |

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
## Week2

This section provides a web application in the form of an interactive Shiny dashboard built using R for analyzing audio emotion inference results.
It allows users to upload inference datasets (CSV or Excel), map emotions into custom classes, and visualize trends over time.

Features
- Upload CSV or Excel inference datasets, that will contain dynamic emotion inference mapping
- Interactive visualizations including class distribution, timelines, daily trends, hour-wise heatmaps, and per-ID analysis
- Dark and light mode toggle
- Time-based filtering for recent data

Dataset Format
The uploaded dataset must contain at least three columns:
1. id – Device or audio identifier
2. timestamp – Time in HH:MM:SS format
3. inference_of_emotion – Predicted emotion label

The time-only values are automatically converted into full timestamps(YYYY-MM-DD HH:MM:SS) inside the application.

Dashboard Tabs Description
Class Distribution:
Shows the number of samples falling into each user-defined class.

Timeline:
Displays emotion or class occurrences over time.

Daily Trend:
Aggregates class counts per day to identify overall emotional trends.

Hour-wise Heatmap:
Visualizes emotional activity distribution across hours of the day.

Per-ID Analysis:
Allows focused analysis on a specific device or audio ID.

How to Run the Application
Install R
Windows / macOS / Linux
1. Go to CRAN (official R website)
    👉 https://cran.r-project.org
    - Download and install R (latest version)
    - During installation, keep all default options

✔ After installation, you should be able to open R or R-GUI

2. Install RStudio (Strongly Recommended)
    RStudio makes running Shiny apps much easier.
    - Go to 👉 https://posit.co/download/rstudio-desktop/
    - Download RStudio Desktop (Free)
    - Install it normally
✔ Open RStudio after installation

3. Clone or Download the Repository
   Option A: Download ZIP (Beginner-friendly)
    - Open the GitHub repository
    - Click Code → Download ZIP
    - Extract the folder to any location
    (e.g., Desktop or Documents)

    Option B: Git (Optional)
    git clone <repository-url>

4. Open the Project in RStudio
    - Open RStudio
    - Click File → Open Folder
    - Select the folder containing app.R

You should now see app.R in the Files pane

5. Install Required R Packages
    In the RStudio Console, run this once:
    install.packages(c(
      "shiny",
      "readxl",
      "readr",
      "dplyr",
      "ggplot2",
      "lubridate",
      "tidyr",
      "bslib"
    ))
📌 Notes:
•	Ignore Rtools warnings (not required for this app)

6. Run the Shiny App
    Method 1 (Recommended)
    - Open app.R
    - Click the Run App button (top-right of editor)
    Method 2 (Console)
    shiny::runApp()
✔ The app will open in your browser at:
    http://127.0.0.1:<port>

✅ System Requirements
•	R ≥ 4.2

Contents of week2 folder:
1. app.R ---> Main Shiny application
2. Basic.R ---> To test if R is installed and shiny package is working
3. tess_emotion_log.xlsx ---> File generated from TESS dataset for input to the app.R
4. synthetic_emotion_inference.xlsx ---> Synthetically generated dataset
5. Convert_dataset_to_excel.py ---> Dataset conversion from TESS dataset to .xlsx file
6. synthetic_data_generation.py ---> Synthetic data creation python script
7. .RData ---> R workspace (auto-generated)
8. .Rhistory ---> R command history for your reference

Here is your **properly structured Markdown section** with clean headers and correctly formatted `bash` and `powershell` code blocks:


# Week 4
## TinySafetyNet – AI Safety Prediction & Spark Analytics Platform

TinySafetyNet is a **Data Engineering + Machine Learning project** that simulates a real-world **women safety monitoring platform**.

The system predicts the **safety level of a location** based on contextual and environmental signals such as:

* Location coordinates
* Time of day
* Crowd density
* Lighting conditions
* Noise levels
* Emotional signals
* Risk scores

The project includes:

• Synthetic data generation
• Machine learning safety prediction
• Real-time monitoring pipeline
• Apache Spark analytics pipeline
• OLAP cube generation
• Interactive dashboard visualization

This repository demonstrates a **complete data pipeline from data generation → ML inference → big data analytics → dashboard visualization**.

---

# System Architecture

```
Synthetic Data Generation
        │
        ▼
synthetic_safety_dataset.csv
        │
        ▼
Machine Learning Model
(women_safety_dscnn_f16.tflite)
        │
        ▼
Real-Time Monitoring Pipeline
        │
        ▼
SQLite Databases
(safety_data.db / safety_logs.db)
        │
        ▼
Apache Spark Processing
        │
        ▼
OLAP Cubes
        │
        ▼
Interactive Dashboard
```

---

# Core Components

The project is divided into **two major modules**:

1️⃣ **AI Safety Prediction System**
2️⃣ **Apache Spark Analytics Pipeline**

---

# 1️⃣ AI Safety Prediction System

This module simulates a **real-time safety prediction system** using a trained AI model.

## Key Features

• Synthetic geospatial dataset generation
• AI-based safety classification
• Real-time safety prediction
• Database logging
• Streamlit dashboard

---

# Application Layer

### app.py

Main **Streamlit dashboard application** that displays safety predictions and analytics.

### realtime_app.py

Simulates **live safety monitoring**, generating real-time predictions using the AI model.

### View.py

Handles dashboard **UI layout, visualization, and map rendering**.

---

# Model Layer

### women_safety_dscnn_f16.tflite

Trained **TensorFlow Lite model** used for safety prediction.

### model_utils.py

Utility functions for:

* Loading the model
* Feature preprocessing
* Running predictions
* Decoding prediction results

### classes.npy

Stores label mappings used by the model.

Example:

```
0 → safe
1 → unsafe
2 → danger
```

---

# Data Generation Layer

### generate_synthetic_dataset.py

Creates synthetic data simulating safety-related environmental signals.

### SynDataGen1Mil.py

Generates **large-scale datasets (1M+ rows)** used for big data processing experiments.

### synthetic_safety_dataset.csv

Dataset used to train and test the safety prediction model.

---

# Data Collection Layer

### data_collector.py

Collects incoming prediction data and sends it to the database.

---

# Database Layer

### db_manager.py

Handles database operations such as:

* Creating tables
* Inserting safety records
* Querying historical data

### safety_data.db

Stores safety prediction records.

### safety_logs.db

Stores system logs and monitoring events.

---

# Configuration

### config.py

Centralized configuration file containing:

* database paths
* model paths
* system parameters

---

# Testing

### test.py

Utility script used to test model predictions and database functionality.

---

# 2️⃣ Apache Spark Analytics Pipeline

This module demonstrates **large-scale analytics using Apache Spark**.

It processes the generated safety data and performs:

* clustering
* time analysis
* emotion analysis
* risk scoring
* streaming analytics
* OLAP cube generation

---

# Spark Query Modules

These scripts perform various **Spark transformations and analytics tasks**.

### clustering.py

Performs clustering to identify **high-risk geographic zones**.

### danger_count.py

Counts the number of **danger-level events** across regions.

### emotion_analysis.py

Analyzes emotional signals associated with safety incidents.

### generate_datacubes.py

Generates **OLAP cubes** from processed data.

### metadata.py

Manages metadata definitions used in Spark processing.

### overview.py

Produces high-level summaries of safety trends.

### parquet_conversion.py

Converts raw datasets into **Parquet format for efficient analytics**.

### partition_demo.py

Demonstrates **data partitioning strategies** in Spark.

### performance_compare.py

Compares performance between different data processing methods.

### risk_score.py

Calculates safety risk scores for different locations.

### sql_vs_spark.py

Compares **traditional SQL queries vs Spark queries**.

### streaming.py

Simulates **real-time streaming data processing**.

### summary.py

Creates aggregated summaries for analytics dashboards.

### time_analysis.py

Analyzes safety patterns based on time-of-day and hour distributions.

---

# OLAP Cube System

The project builds **multidimensional OLAP cubes** for advanced analytics.

## OLAP Cube Types

### risk_score_cube

Stores aggregated safety risk scores.

### geographic_cube

Aggregates safety data by geographic regions.

### emotion_risk_cube

Combines emotional signals with risk scores.

### hour_risk_cube

Analyzes risk by time-of-day.

### hour_emotion_risk_cube

Combines hourly and emotional analysis.

---

# OLAP Query Modules

These scripts perform **OLAP queries on the generated cubes**.

### olap_cube_visualizer.py

Visualizes OLAP cube data.

### olap_emotion_risk.py

Analyzes emotional signals across risk categories.

### olap_geographic_risk.py

Shows geographic distribution of safety risks.

### olap_peak_hours.py

Identifies high-risk hours during the day.

### olap_risk_score.py

Analyzes aggregated safety risk scores.

### olap_time_emotion.py

Examines relationships between time and emotional signals.

---

# Dashboard Module

### dashboard.py

Interactive analytics dashboard displaying:

* safety trends
* risk distributions
* geographic patterns
* OLAP analytics

### styles.css

Custom styling for the dashboard UI.

### header.png

Header image used in the dashboard interface.

---

# Environment Setup

### environment.yml

Conda environment configuration for Spark analytics.

### requirements.txt

Python dependencies required for the AI safety prediction module.

---

# Installation

## Using Conda (Recommended)

```
conda env create -f safety_ai_environment.yml
conda activate safety_ai
```

---

## Using Pip

```
pip install -r requirements.txt
```

---

# Running the Application

Start the main dashboard:

```
streamlit run app.py
```

Run the Spark analytics dashboard:

```
streamlit run dashboard.py
```

---

# Dataset Generation

Generate a dataset:

```
python generate_synthetic_dataset.py
```

Generate large-scale dataset:

```
python SynDataGen1Mil.py
```

---

# Technologies Used

Python
Streamlit
TensorFlow Lite
NumPy
Pandas
SQLite
Apache Spark
Parquet
OLAP Analytics


---
# Running the Application and Technology Specifics

Source data can be **CSV files, Parquet files, or a database (db)**.
Parquet is preferred because it is **column-based**, and converting CSV to Parquet reduces dataset loading and processing time.

---

## 1. Install OpenJDK17 LTS MSI win64x

https://adoptium.net/download?link=https%3A%2F%2Fgithub.com%2Fadoptium%2Ftemurin17-binaries%2Freleases%2Fdownload%2Fjdk-17.0.18%252B8%2FOpenJDK17U-jdk_x64_windows_hotspot_17.0.18_8.msi&vendor=Adoptium

Use **default settings** during installation.
The installer will automatically add Java to the system PATH.

---

## 2. Verify Java Installation

In the terminal run:

```
java --version
```

It should show:

```
openjdk version "17.0.18" 2026-01-20
OpenJDK Runtime Environment Temurin-17.0.18+8 (build 17.0.18+8)
OpenJDK 64-Bit Server VM Temurin-17.0.18+8 (build 17.0.18+8, mixed mode, sharing)
```

---

## 3. Conda Environments

If you're using **Anaconda Prompt or Miniconda**, it is easier to build environments.

`safety_ai` is the environment used to run the code under the **Week4 directory**.

`women_safety_ai` is the environment used for **Apache Spark**.

In CMD, inside the directory, run:

```
conda activate safety_ai
```

or

```
conda activate women_safety_ai
```

---

## 4. Check Spark Installation

Run:

```
pyspark
```

Then check the Python path:

```
where python
```

Set the PySpark Python paths:

```
set PYSPARK_PYTHON=C:\Users\Vibhav\miniconda3\envs\safety_ai\python.exe
```

```
set PYSPARK_DRIVER_PYTHON=C:\Users\Vibhav\miniconda3\envs\safety_ai\python.exe
```

Test Spark:

```
>>>spark.range(10).show()
```

Exit PySpark:

```
>>> exit()
```

---

## 5. Generate Dataset

Generate a dataset of **1 million records** by running:

```
SynDataGen1Mil.py
```

---

## 6. Run Dashboard

`dashboard.py` allows the user to run queries on the dataset.

---

## Queries That Can Be Run

### 0. Parquet Conversion

Convert the dataset to **Parquet format** and save it separately.

---

### 1. Dataset Distribution

Show how Spark divides the dataset into **distributed partitions**.

#### 1.1 Spark vs Parquet

Compare Spark processing using CSV and Parquet.
Parquet performs better for clustering and analytics.

---

### 2. SQL vs Spark Query Time

Apply **row count with SQL** and **row count with Spark**, and compare the time taken by the two approaches.

---

### 3. Dataset Overview

Provide a high-level overview of the dataset.

---

### 4. Danger Event Count

Calculate the **total count of danger events**.

---

### 5. Emotion Count Analysis

Perform **emotion count analysis** on the dataset.

---

### 6. Time-Based Risk Analysis

Analyze safety risk based on **time of day**.
This must be shown **graphically**.

---

### 7. Geographic Hotspots

Perform **K-Means clustering on all danger events** in the dataset to identify unsafe zones.

We will:

1️⃣ Take danger events only
2️⃣ Use latitude and longitude as features
3️⃣ Run KMeans clustering
4️⃣ Discover unsafe zones automatically

The resulting coordinates will be plotted on a map to identify high-risk areas.

---

### 8. Risk Score Calculation

Calculate safety **risk scores** based on aggregated data.

---

### 9. Summary Report

Generate a **summary report** and save it as a CSV file.

---

### 10. Real-Time Data Simulation

Simulate real-time data using:

```
stream_generator.py
```

This stream updates the **K-Means algorithm** continuously.

It generates **real-time safety alerts**, producing signals and blinking danger spots on the map.

The map is restricted to **NSUT coordinates**, using latitude and longitude values within the NSUT region for this simulation.

The generated data is stored in:

```
live.db
```

---

### 11. Combined Queries and OLAP Operations

Execute combined queries and **OLAP-style operations** on the dataset.

---

## Environment Conflict

If two environments exist in the same directory, Python may access the **upper environment** by default.

To force the correct environment, run:

```
python -m streamlit run dashboard.py
```

---

# Hadoop Setup for Parquet

Hadoop is required for Parquet support on Windows.

Create a Hadoop folder in the **C drive**.

---

## Step 1 — Download the ZIP

Open this page:

https://github.com/cdarlint/winutils

Click:

```
Code → Download ZIP
```

You will get a file like:

```
winutils-master.zip
```

---

## Step 2 — Extract the ZIP

Extract it somewhere like:

```
C:\temp\winutils-master
```

Inside you will see folders like:

```
hadoop-3.3.6
hadoop-3.3.5
hadoop-3.2.x
```

Open:

```
hadoop-3.3.6
```

Inside it there is a folder:

```
bin
```

---

## Step 3 — Copy the bin Folder

Copy the entire **bin folder** to:

```
C:\hadoop
```

So the final structure becomes:

```
C:\hadoop
   └── bin
        ├── winutils.exe
        ├── hadoop.dll
        ├── hdfs.dll
        └── other files
```

Important: copy the folder itself, not individual files.

---

## Step 4 — Set Environment Variable

Open CMD and run:

```
setx HADOOP_HOME C:\hadoop
```

Then update PATH:

```
setx PATH "%PATH%;C:\hadoop\bin"
```

---

## Step 5 — Restart Terminal

Close CMD and open a new one.

Verify:

```
echo %HADOOP_HOME%
```

You should see:

```
C:\hadoop
```

Test:

```
winutils.exe
```

You should see usage instructions.

---

## Step 6 — Run Spark Again

Go to your project:

```
cd C:\Users\Vibhav\Desktop\tinySafetyNet\Week4\project
```

Activate the environment inside the project directory:

```
conda activate women_safety_ai
```

Run the dashboard:

```
python -m streamlit run dashboard.py
```

---

Note: while using `setx`, you may encounter overwriting problems that could overwrite your previous PATH variable, so be careful.


TinySafetyNet/
│
├── app.py
├── realtime_app.py
├── View.py
├── config.py
├── test.py
│
├── model_utils.py
├── classes.npy
├── women_safety_dscnn_f16.tflite
│
├── generate_synthetic_dataset.py
├── SynDataGen1Mil.py
├── synthetic_safety_dataset.csv
│
├── data_collector.py
├── db_manager.py
│
├── safety_data.db
├── safety_logs.db
│
├── requirements.txt
├── safety_ai_environment.yml
│
├── project/
│   │
│   ├── dashboard.py
│   ├── environment.yml
│   ├── header.png
│   ├── live.db
│   ├── styles.css
│   │
│   ├── queries/
│   │   ├── clustering.py
│   │   ├── danger_count.py
│   │   ├── emotion_analysis.py
│   │   ├── generate_datacubes.py
│   │   ├── metadata.py
│   │   ├── overview.py
│   │   ├── parquet_conversion.py
│   │   ├── partition_demo.py
│   │   ├── performance_compare.py
│   │   ├── risk_score.py
│   │   ├── sql_vs_spark.py
│   │   ├── streaming.py
│   │   ├── summary.py
│   │   └── time_analysis.py
│   │
│   └── olap_cubes/
│       │
│       ├── olap_cube_visualizer.py
│       ├── olap_emotion_risk.py
│       ├── olap_geographic_risk.py
│       ├── olap_peak_hours.py
│       ├── olap_risk_score.py
│       ├── olap_time_emotion.py
│       │
│       ├── emotion_risk_cube/
│       ├── geographic_cube/
│       ├── hour_emotion_risk_cube/
│       ├── hour_risk_cube/
│       └── risk_score_cube/




---

# ☸️ Kubernetes Setup (Recommended Production Mode)

This section explains how to run the entire TinySafetyNET ecosystem locally using **Minikube**.

---

## 🔹 Step 1: Prerequisites

Make sure the following tools are installed:

### 1️⃣ Docker Desktop  
- Install and keep it **running in the background**

### 2️⃣ Minikube

```powershell
winget install Kubernetes.minikube
````

### 3️⃣ Kubectl

```powershell
winget install Kubernetes.kubectl
```

---

## 🔹 Step 2: Start Cluster & Connect Docker

Start your local Kubernetes cluster and point your terminal to Minikube’s internal Docker daemon.

```powershell
# Start Minikube cluster
minikube start --driver=docker

# Point your terminal to Minikube's Docker environment
& minikube -p minikube docker-env | Invoke-Expression
```

This allows Docker images to be built directly inside Minikube without pushing them to Docker Hub.

---

## 🔹 Step 3: Build Container Images

Build the two microservice containers:

```bash
# Build Python Streamlit (Inference Service)
docker build -t tinysafety-streamlit:v1 ./week1/streamlit-int8-app

# Build R Shiny (Analytics Service)
docker build -t tinysafety-shiny:v1 ./week2
```

> ⚠️ Note: The Shiny build may take a few minutes due to Linux dependency installation.

---

## 🔹 Step 4: Deploy to Kubernetes

Apply the Kubernetes deployment manifests:

```bash
kubectl apply -f k8s/streamlit-deployment.yaml
kubectl apply -f k8s/shiny-deployment.yaml
```

Verify the pods are running:

```bash
kubectl get pods
```

Wait until the `STATUS` shows:

```
Running
```

---

## 🔹 Step 5: Launch the Dashboards

Expose the services and open them in your browser:

```bash
# Open Streamlit Live Inference Dashboard
minikube service streamlit-service

# Open R Shiny Analytics Dashboard
minikube service shiny-service

# Open Kubernetes Control Dashboard
minikube dashboard
```

---

## ✅ System Ready

Once both dashboards are accessible in the browser and pods show `Running`, your **TinySafetyNET Kubernetes cluster is fully operational**.

```

---
