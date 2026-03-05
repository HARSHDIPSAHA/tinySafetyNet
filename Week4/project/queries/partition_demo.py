import streamlit as st
import pandas as pd


def run_partition_demo(df):

    st.header("Spark Partition Visualization")

    st.markdown("""
This module demonstrates **how Apache Spark distributes datasets into partitions**.

Spark divides a large dataset into multiple partitions so that each partition
can be processed **in parallel by different CPU cores**.

This enables **distributed processing and faster analytics**.
""")

    # current partition count
    partitions = df.rdd.getNumPartitions()

    st.subheader("Current Number of Partitions")

    st.write("Partitions:", partitions)

    # records per partition
    partition_sizes = df.rdd.mapPartitions(lambda x: [sum(1 for _ in x)]).collect()

    partition_df = pd.DataFrame({
        "Partition": range(len(partition_sizes)),
        "Records": partition_sizes
    })

    st.subheader("Records Per Partition")

    st.dataframe(partition_df)

    # GRAPH
    st.subheader("Partition Distribution (Graph)")

    st.bar_chart(partition_df.set_index("Partition"))

    # repartition section
    st.subheader("Repartition Dataset")

    new_partitions = st.slider(
        "Select number of partitions",
        min_value=1,
        max_value=20,
        value=partitions
    )

    if st.button("Repartition Dataset"):

        df_repartitioned = df.repartition(new_partitions)

        st.write("New partition count:", df_repartitioned.rdd.getNumPartitions())

        new_sizes = df_repartitioned.rdd.mapPartitions(lambda x: [sum(1 for _ in x)]).collect()

        repartition_df = pd.DataFrame({
            "Partition": range(len(new_sizes)),
            "Records": new_sizes
        })

        st.subheader("New Partition Distribution")

        st.dataframe(repartition_df)

        st.bar_chart(repartition_df.set_index("Partition"))