import streamlit as st
import pandas as pd

def run_olap_emotion_risk(spark):

    st.header("OLAP Analysis: Emotion vs Risk")

    cube = spark.read.parquet("olap_cubes/emotion_risk_cube")

    result = cube.toPandas()

    st.dataframe(result)

    pivot = result.pivot(
        index="emotion",
        columns="risk_level",
        values="count"
    ).fillna(0)

    st.subheader("Emotion vs Risk Visualization")

    st.bar_chart(pivot)