from pyspark.sql.dataframe import DataFrame as DF
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.types import IntegerType
from pyspark.rdd import RDD

# Eucledian distance
def dist(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def plotDistribution(rdd: RDD):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    points = rdd.collect()
    distances = [point[1][2] for point in points]
    plt.hist(distances, bins=100)
    plt.show()
    

"""_summary_
We are using the kmeans calculation from the previous step, before merging the clusters.
We use it to calculate the distance between each point and it's cluster center.
We then calculate the average distance for each cluster and the standard deviation.
We then split the points into two groups, outliers and non-outliers, based on their deviation above the average distance to the centers
Since we know from using k-means that the points inside them are more or less evenly distributed, we can use the standard deviation as a threshold to find outliers in O(n) time!.
"""
def findOutliers(df: DF, n_stdev: int):
    """df
    +--------------------+-------------------+-------+                              
    |                 _c0|                _c1|cluster|
    +--------------------+-------------------+-------+
    |  0.8095605242868158| 0.9369266055045872|    110|
    | 0.09348496530454897|  0.944954128440367|     50|
                ...             ...             ... 
    """
    """ reducedClusters
    +--------------------+-------------------+-------+
    |                 _c0|                _c1|cluster|
    +--------------------+-------------------+-------+
    |  0.8095605242868158| 0.9369266055045872|      0|
    | 0.09348496530454897|  0.944954128440367|      1|
                ...             ...             ... 
    """
    
    # Adding index to the dataframe
    df = df.withColumn("index", F.monotonically_increasing_id())
    
    # Converting the dataframe rows into a RDD of tuples
    indexedClusteredRDD = df.rdd.map(lambda x: ((x[3], x[2]), (x[0], x[1])))
    """List of: For all points
    ((index, cluster), (_c0, _c1))
    """
    
    # Creating a lookup table for the centers of the clusters
    centers = df.groupBy("cluster").avg("_c0", "_c1").withColumnRenamed("avg(_c0)", "_c0").withColumnRenamed("avg(_c1)", "_c1").collect()
    centerLUT = {}
    for center in centers:
        centerLUT[center["cluster"]] = (center["_c0"], center["_c1"])
    
    # Calculating the distance between each point and it's cluster center
    clusteredRDD = indexedClusteredRDD.map(lambda x: (x[0][1], dist(x[1], centerLUT[x[0][1]])))
    """list of all points incl. their distances
    (cluster, distance)
    x[0], x[1]
    """
    # print(clusteredRDD.take(5))
    
    # Calculating the average distance for each cluster and storing it in a dict
    # https://stackoverflow.com/a/29930162/9183984
    avgRDD = clusteredRDD.aggregateByKey((0, 0), lambda a,b: (a[0] + b, a[1] + 1), lambda a,b: (a[0] + b[0], a[1] + b[1])).mapValues(lambda v: v[0]/v[1])
    avgRDDLut = avgRDD.collectAsMap()
    
    # Adding the average distance to the RDD so we can perform calculations
    AVGStoredClusteredRdd = clusteredRDD.map(lambda x: (x[0], (x[1], avgRDDLut[x[0]])))
    
    # Calculating the standard deviation for each cluster and storing it in a dict
    stdevRDD = AVGStoredClusteredRdd.aggregateByKey((0, 0), lambda a,b: (a[0] + (b[0] - b[1])**2, a[1] + 1), lambda a,b: (a[0] + b[0], a[1] + b[1])).mapValues(lambda v: (v[0]/v[1])**0.5)
    stdevLUT = stdevRDD.collectAsMap()
    # print(stdevLUT)
    
    
    def isOutlier(x):
        # x = ((index, cluster), (_c0, _c1))
        cluster = x[0][1]
        index = x[0][1]
        point = x[1]
        distance = dist(point, centerLUT[cluster])
        avg = avgRDDLut[cluster]
        stdev = stdevLUT[cluster]
        if distance > avg + n_stdev*stdev:
            return True
        else:
            return False
    
    # Creating a dataframe with the outliers
    outlierDF = indexedClusteredRDD.map(lambda x: (1, x[1][0], x[1][1]) if isOutlier(x) else (0, x[1][0], x[1][1])).toDF(["outlier", "_c0", "_c1"])

    
    return outlierDF