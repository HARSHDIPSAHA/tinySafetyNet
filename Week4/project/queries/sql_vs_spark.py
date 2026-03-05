import streamlit as st
import pandas as pd
import time


def run_sql_vs_spark(spark, df):

    st.header("SQL vs Spark Row Count Comparison")

    st.markdown("""
This experiment compares two methods of querying data in Apache Spark:

• **Spark DataFrame API** – programmatic operations  
• **Spark SQL** – traditional SQL query interface  

Both approaches operate on the same distributed dataset.

We measure **execution time for counting rows** using each method.
""")

    if st.button("Run Comparison"):

        # Spark DataFrame count
        start_spark = time.time()

        spark_count = df.count()

        end_spark = time.time()

        spark_time = end_spark - start_spark

        # Register dataframe as SQL table
        df.createOrReplaceTempView("safety_events")

        # SQL query count
        start_sql = time.time()

        sql_count = spark.sql(
            "SELECT COUNT(*) as total_rows FROM safety_events"
        ).collect()[0]["total_rows"]

        end_sql = time.time()

        sql_time = end_sql - start_sql

        # results table
        results = pd.DataFrame({
            "Method": ["Spark DataFrame API", "Spark SQL"],
            "Row Count": [spark_count, sql_count],
            "Execution Time (seconds)": [spark_time, sql_time]
        })

        st.subheader("Results")

        st.dataframe(results)

        # graph
        st.subheader("Execution Time Comparison")

        chart_data = results[["Method", "Execution Time (seconds)"]]

        st.bar_chart(chart_data.set_index("Method"))

        # explanation
        st.markdown("""
Both methods operate on Spark's distributed execution engine.  
The difference is only **how the query is expressed**:

• DataFrame API → programmatic operations  
• SQL → declarative queries

Internally, Spark optimizes both using the **Catalyst Query Optimizer**.
""")