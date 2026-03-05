import streamlit as st
import pandas as pd

def run_olap_peak_hours(spark):

    st.header("OLAP Analysis: Peak Danger Hours")

    cube = spark.read.parquet("olap_cubes/hour_risk_cube")

    danger = cube.filter(cube.risk_level == "🚨 DANGER") \
        .orderBy("count", ascending=False)

    result = danger.toPandas()

    st.subheader("Peak Danger Hours")

    st.dataframe(result)

    chart = result[["hour", "count"]].set_index("hour")

    st.bar_chart(chart)