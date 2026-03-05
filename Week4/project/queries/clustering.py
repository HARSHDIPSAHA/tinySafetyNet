
import streamlit as st
import pandas as pd
import folium
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans


from streamlit.components.v1 import html


def run_clustering(df):

    st.header("Geographic Hotspot Detection (KMeans Clustering)")

    st.markdown("""
This module automatically detects **unsafe geographic zones** using machine learning.

Steps performed:

1️⃣ Filter **danger events**  
2️⃣ Use **latitude and longitude as features**  
3️⃣ Apply **KMeans clustering**  
4️⃣ Detect cluster centers representing **high-risk zones**
""")

    # ===============================
    # Filter danger events
    # ===============================

    danger_df = df.filter(df.risk_level == "🚨 DANGER")

    st.write("Total danger events:", danger_df.count())

    # ===============================
    # Create feature vectors
    # ===============================

    assembler = VectorAssembler(
        inputCols=["latitude", "longitude"],
        outputCol="features"
    )

    vector_df = assembler.transform(danger_df)

    # ===============================
    # Choose number of clusters
    # ===============================

    k = st.slider("Select number of clusters", 2, 20, 10)

    if st.button("Run KMeans Clustering"):

        kmeans = KMeans(k=k, seed=1)

        model = kmeans.fit(vector_df)

        centers = model.clusterCenters()

        centers_df = pd.DataFrame(
            centers,
            columns=["Latitude", "Longitude"]
        )

        st.subheader("Detected Hotspot Coordinates")

        st.dataframe(centers_df)

        # ===============================
        # Create Map
        # ===============================

        map_center = [28.6139, 77.2090]

        m = folium.Map(location=map_center, zoom_start=11)

        for i, row in centers_df.iterrows():

            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=f"Hotspot {i}",
                icon=folium.Icon(color="red")
            ).add_to(m)

        st.subheader("Hotspot Map")

        html(m._repr_html_(), height=500)