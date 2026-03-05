import streamlit as st
import pandas as pd

def run_olap_risk_score(spark):

    st.header("OLAP Analysis: Risk Score Cube")

    cube = spark.read.parquet("olap_cubes/risk_score_cube")

    result = cube.orderBy("risk_score", ascending=False).toPandas()

    st.subheader("Highest Risk Conditions")

    st.dataframe(result.head(20))

    chart = result.head(10)[["emotion","risk_score"]].set_index("emotion")

    st.bar_chart(chart)