from helpers import cleanup, normalize
import pyspark.sql.dataframe as DF
from pyspark import SparkContext as sc

from pyspark.sql import SparkSession

if __name__ == "__main__":
    
    spark: SparkSession = SparkSession.builder.appName("dataminingProject").getOrCreate()
    
    df: DF = spark.read.csv("data-example2224.csv")
    
    # TODO: implement
    clean_df = cleanup(df)
    normalized_df = normalize(df)
    
    print("Hello World!")