import streamlit as st
import pandas as pd
from pyspark.sql.functions import hour, to_timestamp


def run_time_analysis(df):

    st.header("Time-Based Risk Analysis")

    st.markdown("""
This module analyzes how **safety incidents vary across different hours of the day**.

By extracting the **hour from the timestamp**, we can observe patterns in when
dangerous situations are most likely to occur.

This helps identify **high-risk time periods**.
""")

    # ensure timestamp is treated as timestamp
    df_time = df.withColumn(
        "timestamp_parsed",
        to_timestamp("timestamp")
    )

    df_time = df_time.withColumn(
        "hour",
        hour("timestamp_parsed")
    )

    # group by hour and risk
    hourly_risk = df_time.groupBy("hour", "risk_level") \
        .count() \
        .orderBy("hour")

    hourly_pd = hourly_risk.toPandas()

    st.subheader("Hourly Risk Distribution Table")

    st.dataframe(hourly_pd)

    # pivot for visualization
    pivot = hourly_pd.pivot(
        index="hour",
        columns="risk_level",
        values="count"
    ).fillna(0)

    st.subheader("Risk Distribution by Hour")

    st.line_chart(pivot)

    # identify peak danger hour
    danger_hour = hourly_pd[hourly_pd["risk_level"] == "🚨 DANGER"]

    if not danger_hour.empty:
        peak_hour = danger_hour.loc[danger_hour["count"].idxmax()]["hour"]
        st.success(f"Peak danger hour detected: {int(peak_hour)}:00")