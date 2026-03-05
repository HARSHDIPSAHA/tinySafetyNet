'''
import streamlit as st
import pandas as pd
import plotly.express as px


def run_olap_cube_visualizer(spark):

    st.header("3D OLAP Cube Explorer")

    st.markdown("""
This module visualizes the **multidimensional OLAP cube**.

Dimensions used:

• Hour of day  
• Emotion  
• Risk level  

Measure:

• Event count

You can rotate the cube to explore relationships between
time, emotion, and safety risk.
""")

    cube = spark.read.parquet("olap_cubes/hour_emotion_risk_cube")

    df = cube.toPandas()

    st.subheader("OLAP Cube Data")

    st.dataframe(df.head(50))

    # 3D scatter cube
    fig = px.scatter_3d(
        df,
        x="hour",
        y="emotion",
        z="count",
        color="risk_level",
        size="count",
        title="3D OLAP Cube: Time vs Emotion vs Risk"
    )

    st.plotly_chart(fig, use_container_width=True)'''

import streamlit as st
import pandas as pd
import plotly.express as px


def run_olap_cube_visualizer(spark):

    st.header("3D OLAP Cube Explorer")

    st.markdown("""
This module visualizes the **multidimensional OLAP cube**.

Dimensions used:

• Hour of day  
• Emotion  
• Risk level  

Measure:

• Event count

You can rotate the cube to explore relationships between
time, emotion, and safety risk.
""")

    # Ensure Spark session exists
    if spark is None:
        st.error("Spark session not initialized.")
        return

    try:
        cube = spark.read.parquet("olap_cubes/hour_emotion_risk_cube")
    except Exception as e:
        st.error(f"Could not load OLAP cube: {e}")
        return

    # Convert to pandas
    df = cube.toPandas()

    if df.empty:
        st.warning("OLAP cube is empty.")
        return

    # Ensure correct datatypes
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")

    st.subheader("OLAP Cube Data")
    st.dataframe(df.head(50))

    # 3D scatter cube visualization
    fig = px.scatter_3d(
        df,
        x="hour",
        y="emotion",
        z="count",
        color="risk_level",
        size="count",
        title="3D OLAP Cube: Time vs Emotion vs Risk",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)