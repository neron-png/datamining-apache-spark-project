from helpers import cleanup, normalize, plot, testing_listPlot
from pyspark.sql.dataframe import DataFrame as DF
from pyspark import SparkContext as sc
from pyspark.sql import SparkSession
from kmeans import kmeansCluster, reduceClusters, single_link
from outlierDetect import findOutliers

if __name__ == "__main__":
    
    spark: SparkSession = SparkSession.builder.appName("dataminingProject").master("local[*]").getOrCreate()
    df: DF = spark.read.csv("./data-example2223.csv", header=False)
    
    # PART 2
    clean_df = cleanup(df)
    
    # PART 3
    normalized_df = normalize(df)
    
    # PART 4
    clustered = kmeansCluster(normalized_df, n=150)
    clustered.show()

    # Remember to install pandas and matplotlib
    plot(clustered, "cluster")
    
    # Reduce clusters to 5
    reduceClusters = single_link(clustered, n=5)
    plot(reduceClusters, "realCluster")
    
    
    # PART 5
    outlierDF = findOutliers(clustered, n_stdev=5)
    outlierDF.show()
    plot(outlierDF, "outlier")
    