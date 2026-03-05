import streamlit as st
import pandas as pd


def run_emotion_analysis(df):

    st.header("Emotion Analysis")

    st.markdown("""
This module analyzes the **distribution of emotions detected in safety events**.

The emotion signals come from the audio processing pipeline and may indicate
the emotional state of a person during the event.

Understanding emotion patterns can help identify signals associated with
**dangerous or stressful situations**.
""")

    # ==================================
    # Emotion distribution
    # ==================================

    emotion_counts = df.groupBy("emotion") \
        .count() \
        .orderBy("count", ascending=False)

    emotion_pd = emotion_counts.toPandas()

    st.subheader("Emotion Frequency Distribution")

    st.dataframe(emotion_pd)

    st.subheader("Emotion Distribution Graph")

    st.bar_chart(emotion_pd.set_index("emotion"))

    # ==================================
    # Emotion vs Risk
    # ==================================

    st.subheader("Emotion vs Risk Level")

    emotion_risk = df.groupBy("emotion", "risk_level") \
        .count() \
        .orderBy("emotion")

    emotion_risk_pd = emotion_risk.toPandas()

    st.dataframe(emotion_risk_pd)

    # Pivot for visualization
    pivot = emotion_risk_pd.pivot(
        index="emotion",
        columns="risk_level",
        values="count"
    ).fillna(0)

    st.subheader("Emotion vs Risk Visualization")

    st.bar_chart(pivot)