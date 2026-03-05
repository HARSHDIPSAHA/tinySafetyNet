from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, to_timestamp, col

spark = SparkSession.builder \
    .appName("WomenSafetyOLAP") \
    .master("local[*]") \
    .getOrCreate()

# load dataset
df = spark.read.csv(
    "data/women_safety_1M_dataset.csv",
    header=True,
    inferSchema=True
)

# parse timestamp
df = df.withColumn(
    "timestamp_parsed",
    to_timestamp("timestamp")
)

df = df.withColumn(
    "hour",
    hour("timestamp_parsed")
)

# ============================
# Cube 1 : Hour vs Risk Level
# ============================

cube1 = df.groupBy("hour", "risk_level").count()

cube1.write.mode("overwrite").parquet(
    "olap_cubes/hour_risk_cube"
)

# ============================
# Cube 2 : Emotion vs Risk
# ============================

cube2 = df.groupBy("emotion", "risk_level").count()

cube2.write.mode("overwrite").parquet(
    "olap_cubes/emotion_risk_cube"
)

# ============================
# Cube 3 : Geographic Cube
# ============================

geo_df = df.withColumn(
    "lat_bucket",
    (col("latitude") * 10).cast("int")
).withColumn(
    "lon_bucket",
    (col("longitude") * 10).cast("int")
)

cube3 = geo_df.groupBy(
    "lat_bucket",
    "lon_bucket",
    "risk_level"
).count()

cube3.write.mode("overwrite").parquet(
    "olap_cubes/geographic_cube"
)

# ============================
# Cube 4 : Hour + Emotion + Risk
# ============================

cube4 = df.groupBy(
    "hour",
    "emotion",
    "risk_level"
).count()

cube4.write.mode("overwrite").parquet(
    "olap_cubes/hour_emotion_risk_cube"
)

# ============================
# Cube 5 : Emotion + Hour Risk Score
# ============================

danger_df = df.filter(df.risk_level == "🚨 DANGER")

total = df.groupBy("emotion", "hour").count() \
    .withColumnRenamed("count", "total_events")

danger = danger_df.groupBy("emotion", "hour").count() \
    .withColumnRenamed("count", "danger_events")

risk_cube = total.join(
    danger,
    ["emotion", "hour"],
    "left"
).fillna(0)

risk_cube = risk_cube.withColumn(
    "risk_score",
    col("danger_events") / col("total_events")
)

risk_cube.write.mode("overwrite").parquet(
    "olap_cubes/risk_score_cube"
)

print("All OLAP cubes generated successfully")