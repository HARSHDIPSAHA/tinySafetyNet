import streamlit as st
import pandas as pd


def run_risk_score(df):

    st.header("Risk Score Calculation")

    st.markdown("""
This module calculates a **risk score** based on the proportion of dangerous
events relative to the total number of events.

Risk Score Formula:

Risk Score = Danger Events / Total Events

This helps identify which **conditions or emotional signals are associated
with higher safety risks**.
""")

    # ===============================
    # Total events per emotion
    # ===============================

    total_emotion = df.groupBy("emotion") \
        .count() \
        .withColumnRenamed("count", "total_events")

    # ===============================
    # Danger events per emotion
    # ===============================

    danger_emotion = df.filter(df.risk_level == "🚨 DANGER") \
        .groupBy("emotion") \
        .count() \
        .withColumnRenamed("count", "danger_events")

    # ===============================
    # Join both tables
    # ===============================

    risk_df = total_emotion.join(
        danger_emotion,
        on="emotion",
        how="left"
    ).fillna(0)

    # ===============================
    # Calculate risk score
    # ===============================

    risk_df = risk_df.withColumn(
        "risk_score",
        risk_df["danger_events"] / risk_df["total_events"]
    )

    risk_pd = risk_df.orderBy("risk_score", ascending=False).toPandas()

    st.subheader("Risk Score Table")

    st.dataframe(risk_pd)

    # ===============================
    # Visualization
    # ===============================

    st.subheader("Risk Score Visualization")

    chart = risk_pd[["emotion", "risk_score"]].set_index("emotion")

    st.bar_chart(chart)

    # ===============================
    # Highlight highest risk
    # ===============================

    highest = risk_pd.iloc[0]

    st.success(
        f"Highest risk emotion: {highest['emotion']} "
        f"(Risk Score: {highest['risk_score']:.3f})"
    )