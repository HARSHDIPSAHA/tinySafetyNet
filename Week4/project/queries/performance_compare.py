import streamlit as st
import pandas as pd
import time


def run_performance_compare(spark):

    st.header("CSV vs Parquet Performance Comparison")

    st.markdown("""
This experiment compares **dataset loading performance** between **CSV format** and **Parquet format**.

Parquet is a **columnar storage format** used in big data systems because it:

• Reads only required columns  
• Uses compression  
• Improves query speed  

We measure the **time required for Spark to load the dataset**.
""")

    csv_path = "data/women_safety_1M_dataset.csv"
    parquet_path = "data/women_safety.parquet"

    if st.button("Run Performance Test"):

        # CSV load test
        start_csv = time.time()

        df_csv = spark.read.csv(
            csv_path,
            header=True,
            inferSchema=True
        )

        df_csv.count()

        end_csv = time.time()

        csv_time = end_csv - start_csv

        # Parquet load test
        start_parquet = time.time()

        df_parquet = spark.read.parquet(parquet_path)

        df_parquet.count()

        end_parquet = time.time()

        parquet_time = end_parquet - start_parquet

        # results table
        results = pd.DataFrame({
            "Format": ["CSV", "Parquet"],
            "Load Time (seconds)": [csv_time, parquet_time]
        })

        st.subheader("Performance Results")

        st.dataframe(results)

        # graph
        st.subheader("Load Time Comparison")

        st.bar_chart(results.set_index("Format"))

        # explanation
        if parquet_time < csv_time:
            st.success("Parquet loaded faster than CSV.")
        else:
            st.warning("CSV loaded faster (unexpected result).")