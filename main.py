from helpers import cleanup, normalize, plot
import pyspark.sql.dataframe as DF
from pyspark import SparkContext as sc
from pyspark.sql import SparkSession
from kmeans import kmeansCluster, reduceClusters

if __name__ == "__main__":
    
    spark: SparkSession = SparkSession.builder.appName("dataminingProject").master("local[*]").getOrCreate()
    df: DF = spark.read.csv("./data-example2223.csv", header=False)
    
    clean_df = cleanup(df)
    normalized_df = normalize(df)
    
    normalized_df.show()

    clustered = kmeansCluster(normalized_df, n=150)
    clustered.show()

    # Remember to install pandas and matplotlib
    # plot(clustered)
    
    # WIP: Reduce clusters to 5
    reducedClusters = reduceClusters(clustered, n=5)
    reducedClusters.show()
    
    # plot(reducedClusters)