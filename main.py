from helpers import cleanup, normalize
import pyspark.sql.dataframe as DF
from pyspark import SparkContext as sc

from pyspark.sql import SparkSession

if __name__ == "__main__":
    
    spark: SparkSession = SparkSession.builder.appName("dataminingProject").master("local[*]").getOrCreate()
    df: DF = spark.read.csv("./data-example2223.csv", header=False)
    
    clean_df = cleanup(df)
    normalized_df = normalize(df)
    
    normalized_df.show()