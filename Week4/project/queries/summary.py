import streamlit as st
import pandas as pd
import sqlite3
from pyspark.sql.functions import hour, to_timestamp


def run_summary(spark, df):

    st.header("Live Summary Report")

    st.markdown("""
This panel provides a **high-level summary of safety analytics**.

It combines insights from:

• Historical dataset analysis  
• Streaming events arriving in real time  

The summary updates automatically as new events are generated.
""")

    # ===============================
    # Total events
    # ===============================

    total_events = df.count()

    # ===============================
    # Danger events
    # ===============================

    danger_events = df.filter(
        df.risk_level == "🚨 DANGER"
    ).count()

    danger_percentage = (danger_events / total_events) * 100

    # ===============================
    # Emotion analysis
    # ===============================

    emotion_df = df.groupBy("emotion") \
        .count() \
        .orderBy("count", ascending=False)

    top_emotion = emotion_df.first()["emotion"]

    # ===============================
    # Peak danger hour
    # ===============================

    df_time = df.withColumn(
        "timestamp_parsed",
        to_timestamp("timestamp")
    )

    df_time = df_time.withColumn(
        "hour",
        hour("timestamp_parsed")
    )

    danger_hour = df_time.filter(
        df_time.risk_level == "🚨 DANGER"
    ).groupBy("hour") \
     .count() \
     .orderBy("count", ascending=False)

    peak_hour = danger_hour.first()["hour"]

    # ===============================
    # Streaming data check
    # ===============================

    stream_events = 0

    try:
        conn = sqlite3.connect("live.db")
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM events")

        stream_events = cursor.fetchone()[0]

        conn.close()

    except:
        stream_events = 0

    # ===============================
    # Display metrics
    # ===============================

    st.subheader("System Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Events", total_events)
    col2.metric("Danger Events", danger_events)
    col3.metric("Danger Rate", f"{danger_percentage:.2f}%")

    st.divider()

    col4, col5, col6 = st.columns(3)

    col4.metric("Most Common Emotion", top_emotion)
    col5.metric("Peak Danger Hour", f"{peak_hour}:00")
    col6.metric("Streaming Events Received", stream_events)

    # ===============================
    # Summary text
    # ===============================

    st.subheader("System Summary")

    st.write(f"""
Total dataset events processed: **{total_events}**

Danger events detected: **{danger_events}**

Most frequent emotional signal: **{top_emotion}**

Highest risk time period: **{peak_hour}:00**

Real-time events received: **{stream_events}**
""")

    # ===============================
    # Export report
    # ===============================

    summary_data = {
        "total_events": [total_events],
        "danger_events": [danger_events],
        "danger_rate": [danger_percentage],
        "top_emotion": [top_emotion],
        "peak_danger_hour": [peak_hour],
        "streaming_events": [stream_events]
    }

    summary_df = pd.DataFrame(summary_data)

    if st.button("Export Summary Report"):

        summary_df.to_csv("summary_report.csv", index=False)

        st.success("Summary report saved as summary_report.csv")