'''
import streamlit as st
import os
import time
from pyspark.sql import SparkSession


def run_parquet_conveokayrsion(spark, df):

    st.header("CSV to Parquet Conversion")

    st.markdown("""
This tool converts a **CSV dataset into Parquet format**.

Parquet is preferred in **Apache Spark distributed systems** because:

• Columnar storage  
• Faster analytics queries  
• Better compression  
• Efficient clustering operations
""")

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        os.makedirs("data", exist_ok=True)

        csv_path = os.path.join("data", uploaded_file.name)

        # save uploaded file
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File saved to {csv_path}")

        if st.button("Convert to Parquet"):

            start = time.time()

            df_uploaded = spark.read.csv(
                csv_path,
                header=True,
                inferSchema=True
            )

            parquet_name = uploaded_file.name.replace(".csv", ".parquet")
            parquet_path = os.path.join("data", parquet_name)

            df_uploaded.write.mode("overwrite").parquet(parquet_path)

            end = time.time()

            st.success("Parquet conversion completed")

            st.write("Conversion time:", round(end - start, 3), "seconds")

            # file size comparison
            csv_size = os.path.getsize(csv_path) / (1024 * 1024)

            parquet_size = 0
            for root, dirs, files in os.walk(parquet_path):
                for file in files:
                    parquet_size += os.path.getsize(os.path.join(root, file))

            parquet_size = parquet_size / (1024 * 1024)

            st.subheader("File Size Comparison")

            st.write("CSV Size (MB):", round(csv_size, 2))
            st.write("Parquet Size (MB):", round(parquet_size, 2))

            st.write("Compression Ratio:", round(csv_size / parquet_size, 2))
            '''

import streamlit as st
import os
import time


def run_parquet_conversion(spark, df):

    st.header("CSV → Parquet Conversion")

    st.markdown("""
This module converts datasets from **CSV format to Parquet format**.

Parquet is preferred in Spark systems because it provides:

• Columnar storage  
• Faster queries  
• Better compression  
• Efficient analytics
""")

    data_folder = "data"

    # detect csv datasets automatically
    csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

    if not csv_files:
        st.warning("No CSV datasets found inside the data folder.")
        return

    selected_file = st.selectbox(
        "Select Dataset",
        csv_files
    )

    csv_path = os.path.join(data_folder, selected_file)

    parquet_name = selected_file.replace(".csv", ".parquet")
    parquet_path = os.path.join(data_folder, parquet_name)

    st.write("CSV File:", csv_path)
    st.write("Parquet Output:", parquet_path)

    if st.button("Convert to Parquet"):

        start = time.time()

        df_csv = spark.read.csv(
            csv_path,
            header=True,
            inferSchema=True
        )

        df_csv.coalesce(1).write.mode("overwrite").option("compression","snappy").parquet(parquet_path)
        end = time.time()

        st.success("Conversion Complete")

        st.write("Conversion Time:", round(end - start, 2), "seconds")

        # file size comparison
        csv_size = os.path.getsize(csv_path) / (1024 * 1024)

        parquet_size = 0
        for root, dirs, files in os.walk(parquet_path):
            for file in files:
                parquet_size += os.path.getsize(os.path.join(root, file))

        parquet_size = parquet_size / (1024 * 1024)

        st.subheader("File Size Comparison")

        st.write("CSV Size (MB):", round(csv_size, 2))
        st.write("Parquet Size (MB):", round(parquet_size, 2))
        st.write("Compression Ratio:", round(csv_size / parquet_size, 2))