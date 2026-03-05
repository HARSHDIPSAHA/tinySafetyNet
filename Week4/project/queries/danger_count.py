import streamlit as st
import pandas as pd


def run_danger_count(df):

    st.header("Total Danger Events Analysis")

    st.markdown("""
This module analyzes how many safety events are classified as **DANGER**.

It helps estimate the **frequency of high-risk incidents** within the dataset.
""")

    # total events
    total_events = df.count()

    # risk distribution
    risk_counts = df.groupBy("risk_level") \
        .count() \
        .orderBy("count", ascending=False)

    risk_pd = risk_counts.toPandas()

    # danger events
    danger_events = risk_pd.loc[risk_pd["risk_level"] == "🚨 DANGER", "count"].values

    if len(danger_events) > 0:
        danger_events = danger_events[0]
    else:
        danger_events = 0

    danger_percentage = (danger_events / total_events) * 100

    # metrics
    st.subheader("Key Metrics")

    col1, col2 = st.columns(2)

    col1.metric("Total Events", total_events)
    col2.metric("Danger Events", danger_events)

    st.metric("Danger Percentage", f"{danger_percentage:.2f}%")

    # risk distribution table
    st.subheader("Risk Level Distribution")

    st.dataframe(risk_pd)

    # graph
    st.subheader("Risk Level Visualization")

    chart_data = risk_pd.set_index("risk_level")

    st.bar_chart(chart_data)