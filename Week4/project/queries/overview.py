import streamlit as st
import pandas as pd


def run_overview(df):

    st.header("Dataset Overview")

    st.markdown("""
This module performs **dataset profiling**.

Before performing analytics or machine learning, it is important to
understand the **structure and characteristics of the dataset**.

This section shows:

• Dataset size  
• Schema (column names and types)  
• Sample records  
• Basic statistical summary  
""")

    # ===============================
    # Dataset size
    # ===============================

    st.subheader("Dataset Size")

    row_count = df.count()
    column_count = len(df.columns)

    col1, col2 = st.columns(2)

    col1.metric("Total Rows", row_count)
    col2.metric("Total Columns", column_count)

    # ===============================
    # Schema
    # ===============================

    st.subheader("Dataset Schema")

    schema_data = [(field.name, str(field.dataType)) for field in df.schema]

    schema_df = pd.DataFrame(schema_data, columns=["Column Name", "Data Type"])

    st.dataframe(schema_df)

    # ===============================
    # Sample records
    # ===============================

    st.subheader("Sample Records")

    sample = df.limit(10).toPandas()

    st.dataframe(sample)

    # ===============================
    # Basic statistics
    # ===============================

    st.subheader("Statistical Summary")

    stats = df.describe().toPandas()

    st.dataframe(stats)

    # ===============================
    # Missing values check
    # ===============================

    st.subheader("Missing Values Check")

    null_counts = df.select([
        (df[c].isNull().cast("int")).alias(c)
        for c in df.columns
    ]).groupBy().sum().toPandas()

    st.dataframe(null_counts)