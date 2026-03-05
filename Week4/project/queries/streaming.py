import streamlit as st
import sqlite3
import pandas as pd
import folium
from streamlit.components.v1 import html
import random
import math
import time
from datetime import datetime


def generate_random_point(lat, lon, radius_km=1.5):

    radius_deg = radius_km / 111

    u = random.random()
    v = random.random()

    w = radius_deg * math.sqrt(u)
    t = 2 * math.pi * v

    new_lat = lat + w * math.cos(t)
    new_lon = lon + w * math.sin(t)

    return new_lat, new_lon


def run_streaming():

    st.header("Real-Time NSUT Safety Simulation")

    st.markdown("""
This module simulates **real-time safety alerts around NSUT campus**.

The system generates **random danger events within a 1.5 km radius** around NSUT.

Each event:

• Appears on the map  
• Blinks as a danger signal  
• Gets stored in **live.db**
""")

    nsut_lat = 28.6100
    nsut_lon = 77.0360

    refresh = st.slider("Refresh Interval (seconds)", 1, 10, 3)

    if st.button("Start Simulation"):

        conn = sqlite3.connect("live.db")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            timestamp TEXT,
            latitude REAL,
            longitude REAL,
            risk_level TEXT
        )
        """)

        placeholder = st.empty()

        while True:

            lat, lon = generate_random_point(nsut_lat, nsut_lon)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            conn.execute(
                "INSERT INTO events VALUES (?,?,?,?)",
                (timestamp, lat, lon, "🚨 DANGER")
            )

            conn.commit()

            df = pd.read_sql_query("SELECT * FROM events", conn)

            with placeholder.container():

                st.subheader("Recent Events")

                st.dataframe(df.tail(10))

                st.write("Total events generated:", len(df))

                m = folium.Map(
                    location=[nsut_lat, nsut_lon],
                    zoom_start=15
                )

                for _, row in df.iterrows():

                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=8,
                        color="red",
                        fill=True,
                        fill_color="red"
                    ).add_to(m)

                html(m._repr_html_(), height=500)

            time.sleep(refresh)