import streamlit as st
import pandas as pd

def run_olap_time_emotion(spark):

    st.header("OLAP Analysis: Time and Emotion Patterns")

    cube = spark.read.parquet("olap_cubes/hour_emotion_risk_cube")

    result = cube.toPandas()

    st.dataframe(result)

    pivot = result.pivot_table(
        index="hour",
        columns="emotion",
        values="count",
        aggfunc="sum"
    ).fillna(0)

    st.subheader("Emotion Distribution by Hour")

    st.line_chart(pivot)