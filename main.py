from helpers import cleanup, normalize, denormalize, plot, testing_listPlot
from pyspark.sql.dataframe import DataFrame as DF
from pyspark import SparkContext as sc
from pyspark.sql import SparkSession
from kmeans import kmeansCluster, reduceClusters, single_link
from outlierDetect import findOutliers
import time

start_time = time.time()

if __name__ == "__main__":
    
    spark: SparkSession = SparkSession.builder.appName("dataminingProject").master("local[*]").getOrCreate()
    df: DF = spark.read.csv("./data-example2223.csv", header=False)
    
    # PART 2
    clean_df = cleanup(df)
    
    # PART 3
    normalized_df, min_max_values = normalize(df)
    
    # PART 4
    clustered = kmeansCluster(normalized_df, n=150)
    # clustered.show()

    # Remember to install pandas and matplotlib
    # plot(clustered, "cluster")
    
    # Reduce clusters to 5
    reduceClusters = single_link(clustered, n=5)
    # plot(reduceClusters, "realCluster")
    
    
    # PART 5
    outlierDF = findOutliers(clustered, n_stdev=5)
    denormalized_outlierDF = denormalize(outlierDF, min_max_values)
    denormalized_outlierDF.show(n=outlierDF.count(),truncate=False)
    # plot(outlierDF, "outlier")

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")