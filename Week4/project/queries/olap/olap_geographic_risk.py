import streamlit as st
import pandas as pd
import folium
from streamlit.components.v1 import html

def run_olap_geographic_risk(spark):

    st.header("OLAP Analysis: Geographic Risk Zones")

    cube = spark.read.parquet("olap_cubes/geographic_cube")

    danger = cube.filter(cube.risk_level == "🚨 DANGER")

    result = danger.orderBy("count", ascending=False).limit(50).toPandas()

    st.dataframe(result)

    m = folium.Map(location=[28.6139,77.2090], zoom_start=11)

    for _, row in result.iterrows():

        lat = row["lat_bucket"] / 10
        lon = row["lon_bucket"] / 10

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="red",
            fill=True
        ).add_to(m)

    html(m._repr_html_(), height=500)