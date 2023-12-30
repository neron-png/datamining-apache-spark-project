from helpers import cleanup, normalize, plot, testing_listPlot
from pyspark.sql.dataframe import DataFrame as DF
from pyspark import SparkContext as sc
from pyspark.sql import SparkSession
from kmeans import kmeansCluster, reduceClusters, single_link

if __name__ == "__main__":
    
    spark: SparkSession = SparkSession.builder.appName("dataminingProject").master("local[*]").getOrCreate()
    df: DF = spark.read.csv("./data-example2223.csv", header=False)
    
    clean_df = cleanup(df)
    normalized_df = normalize(df)

    clustered = kmeansCluster(normalized_df, n=150)
    clustered.show()

    # Remember to install pandas and matplotlib
    # plot(clustered, "cluster")
    
    reduceClusters = single_link(clustered, n=5)
    plot(reduceClusters, "realCluster")
    
    # reduced_Kmeans = kmeansCluster(clustered.groupBy("cluster").avg("_c0", "_c1").withColumnRenamed("avg(_c0)", "_c0").withColumnRenamed("avg(_c1)", "_c1").drop("cluster"), n=25)
    # reduced_Kmeans.show()
    # plot(reduced_Kmeans)
    
    # WIP: Reduce clusters to 5
    # reducedClusters = single_link(clustered, n=5)
    # # reducedClusters.show()
    
    # testing_listPlot(reducedClusters)