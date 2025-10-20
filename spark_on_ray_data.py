import ray
import raydp
import logging
import os


os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"] = "0.5"

ray.init()

spark = raydp.init_spark(app_name="RayDP Example",
                         num_executors=2,
                         executor_cores=2,
                         executor_memory="4GB")
# Set Spark logging level
spark.sparkContext.setLogLevel("WARN")

# Spark Dataframe to Ray Dataset
df1 = spark.range(0, 1000)
ds1 = ray.data.from_spark(df1)
ds1.show()

# Ray Dataset to Spark Dataframe
# ds2 = ray.data.from_items([{"id": i} for i in range(1000)])
df2 = ds1.to_spark(spark)
df2.show()