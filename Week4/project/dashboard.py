import os
import streamlit as st
import time
from pyspark.sql import SparkSession
#splash = st.empty()

#with splash.container():
#    st.markdown(
#        """
#        <div style="
#            display:flex;
#            justify-content:center;
#            align-items:center;
#            height:100vh;
#            background-color:#0F0F10;
#        ">
#            <img src="header.png" width="900">
#        </div>
#        """,
#        unsafe_allow_html=True
#    )

#time.sleep(0.8)

#splash.empty()
os.environ["HADOOP_HOME"] = ""

# ================================
# PAGE CONFIG
# ================================

st.set_page_config(
    page_title="Delhi NCR Women Safety Analytics",
    layout="wide"
)
#st.image("header.png", use_container_width=True)




st.title("Delhi NCR Women Safety Analytics Dashboard")

st.markdown("""
This dashboard demonstrates a **Distributed Data Processing Pipeline using Apache Spark**.

The system processes **1 million simulated safety events across Delhi NCR** and allows
interactive exploration of the dataset using various **data mining, machine learning,
and OLAP analytics tools**.

Pipeline Demonstrated:

Dataset → Spark Processing → Data Mining → Clustering → OLAP Analytics → Real-time Simulation
""")

# ================================
# START SPARK SESSION
# ================================

@st.cache_resource
def get_spark():

    spark = SparkSession.builder \
        .appName("WomenSafetyDashboard") \
        .master("local[*]") \
        .config("spark.hadoop.fs.file.impl","org.apache.hadoop.fs.LocalFileSystem") \
        .config("spark.sql.sources.commitProtocolClass",
                "org.apache.spark.sql.execution.datasources.SQLHadoopMapReduceCommitProtocol") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

spark = get_spark()

# ================================
# LOAD DATASET
# ================================

@st.cache_resource
def load_dataset():

    df = spark.read.csv(
        "data/women_safety_1M_dataset.csv",
        header=True,
        inferSchema=True
    )

    return df


df = load_dataset()

# ================================
# SIDEBAR QUERY SELECTION
# ================================

st.sidebar.title("Analytics Modules")

query = st.sidebar.radio(
    "Select Analysis",
    [

        # Data Engineering
        "0. Parquet Conversion",

        # Distributed Processing
        "1. Spark Partition Visualization",
        "1.1 CSV vs Parquet Performance",

        # Query Performance
        "2. SQL vs Spark Row Count Comparison",

        # Data Profiling
        "3. Dataset Overview",

        # Risk Analytics
        "4. Total Danger Events",
        "5. Emotion Analysis",
        "6. Time Based Risk Analysis",

        # Machine Learning
        "7. Geographic Hotspots (KMeans Clustering)",

        # Risk Modeling
        "8. Risk Score Calculation",

        # Monitoring Dashboard
        "9. Live Summary Report",

        # Streaming Simulation
        "10. Real-Time Streaming Simulation",

        # ========================
        # OLAP ANALYTICS
        # ========================

        "OLAP – Peak Danger Hours",
        "OLAP – Emotion Risk Analysis",
        "OLAP – Geographic Risk Zones",
        "OLAP – Time Emotion Analysis",
        "OLAP – Risk Score Cube",
        "OLAP – 3D Cube Explorer"
    ]
)

# ================================
# QUERY EXECUTION
# ================================

# QUERY 0
if query == "0. Parquet Conversion":

    from queries.parquet_conversion import run_parquet_conversion
    run_parquet_conversion(spark, df)


# QUERY 1
elif query == "1. Spark Partition Visualization":

    from queries.partition_demo import run_partition_demo
    run_partition_demo(df)


# QUERY 1.1
elif query == "1.1 CSV vs Parquet Performance":

    from queries.performance_compare import run_performance_compare
    run_performance_compare(spark)


# QUERY 2
elif query == "2. SQL vs Spark Row Count Comparison":

    from queries.sql_vs_spark import run_sql_vs_spark
    run_sql_vs_spark(spark, df)


# QUERY 3
elif query == "3. Dataset Overview":

    from queries.overview import run_overview
    run_overview(df)


# QUERY 4
elif query == "4. Total Danger Events":

    from queries.danger_count import run_danger_count
    run_danger_count(df)


# QUERY 5
elif query == "5. Emotion Analysis":

    from queries.emotion_analysis import run_emotion_analysis
    run_emotion_analysis(df)


# QUERY 6
elif query == "6. Time Based Risk Analysis":

    from queries.time_analysis import run_time_analysis
    run_time_analysis(df)


# QUERY 7
elif query == "7. Geographic Hotspots (KMeans Clustering)":

    from queries.clustering import run_clustering
    run_clustering(df)


# QUERY 8
elif query == "8. Risk Score Calculation":

    from queries.risk_score import run_risk_score
    run_risk_score(df)


# QUERY 9
elif query == "9. Live Summary Report":

    from queries.summary import run_summary
    run_summary(spark, df)


# QUERY 10
elif query == "10. Real-Time Streaming Simulation":

    from queries.streaming import run_streaming
    run_streaming()


# ================================
# OLAP MODULES
# ================================

elif query == "OLAP – Peak Danger Hours":

    from queries.olap.olap_peak_hours import run_olap_peak_hours
    run_olap_peak_hours(spark)


elif query == "OLAP – Emotion Risk Analysis":

    from queries.olap.olap_emotion_risk import run_olap_emotion_risk
    run_olap_emotion_risk(spark)


elif query == "OLAP – Geographic Risk Zones":

    from queries.olap.olap_geographic_risk import run_olap_geographic_risk
    run_olap_geographic_risk(spark)


elif query == "OLAP – Time Emotion Analysis":

    from queries.olap.olap_time_emotion import run_olap_time_emotion
    run_olap_time_emotion(spark)


elif query == "OLAP – Risk Score Cube":

    from queries.olap.olap_risk_score import run_olap_risk_score
    run_olap_risk_score(spark)


elif query == "OLAP – 3D Cube Explorer":

    from queries.olap.olap_cube_visualizer import run_olap_cube_visualizer
    run_olap_cube_visualizer(spark)

