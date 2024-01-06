from pyspark.sql.dataframe import DataFrame as DF
from pyspark import SparkContext as sc
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.types import IntegerType
from pyspark.rdd import RDD
import time
import sys


def cleanup(df: DF) -> DF:
    """Drop all rows which contain null values"""
    cleandf = df.dropna(how='any')
    return cleandf

def normalize(df: DF) -> DF:
    """min-max normalization"""
    min0 = float(df.agg(F.min("_c0")).collect()[0][0])
    max0 = float(df.agg(F.max("_c0")).collect()[0][0])
    min1 = float(df.agg(F.min("_c1")).collect()[0][0])
    max1 = float(df.agg(F.max("_c1")).collect()[0][0])

    min_max_values = {"_c0":{},"_c1":{}}
    min_max_values["_c0"]["min"] = min0
    min_max_values["_c0"]["max"] = max0
    min_max_values["_c1"]["min"] = min1
    min_max_values["_c1"]["max"] = max1

    df_normalized = df.withColumn("_c0", (df["_c0"] - min0) / (max0 - min0))\
        .withColumn("_c1", (df["_c1"] - min1) / (max1 - min1))
    
    return df_normalized, min_max_values

def denormalize(df: DF, min_max_values: dict) -> DF:
    """min-max denormalization"""
    min0 = min_max_values["_c0"]["min"]
    max0 = min_max_values["_c0"]["max"]
    min1 = min_max_values["_c1"]["min"]
    max1 = min_max_values["_c1"]["max"]

    df_denormalized = df.withColumn("_c0", df["_c0"] * (max0 - min0) + min0)\
        .withColumn("_c1", df["_c1"] * (max1 - min1) + min1)
    
    return df_denormalized


# def plot(df: DF, featureColumn: str):
#     import matplotlib.pyplot as plt
#     import pandas as pd

#     pdf = df.toPandas()
#     pdf.plot.scatter(x='_c0', y='_c1', c=featureColumn, cmap="viridis")
#     plt.show()

# def testing_listPlot(df: list):
#     import matplotlib.pyplot as plt
#     import pandas as pd
    
#     plt.scatter(x=[_[0] for _ in df], y=[_[1] for _ in df], c=[_[2] for _ in df], cmap="viridis")
#     plt.show()


def kmeansCluster(df: DF, n: int):
    """ K-means clustering with k=n
    Compressing the input data into a vector and then clustering it, 
    outputting the cluster number for each point under the new column cluster.
    """
    vec_assembler = VectorAssembler(inputCols = ["_c0", "_c1"], 
                                    outputCol='features', handleInvalid="skip") 
    
    final_data = vec_assembler.transform(df)
  
    kmeans = KMeans(k=n, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(final_data)
    clustered = model.transform(final_data)
    
    return clustered.select("_c0", "_c1", "cluster")


def dist(a, b):
    """ Euclidean distance between two points as (x,y) tuples/vectors with length 2 """
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def single_link(df: DF, n: int):
    """_summary_
    Single link clustering algorithm, applie on the centers of the previously acquired clusters.
    Run locally, polynomial but n is small (150)"""
    
    """in
    +--------------------+-------------------+-------+                              
    |                 _c0|                _c1|cluster|
    +--------------------+-------------------+-------+
    |  0.8095605242868158| 0.9369266055045872|    110|
    | 0.09348496530454897|  0.944954128440367|     50|
                ...             ...             ... 
    """
    # Extracting the centers
    centers = df.groupBy("cluster").avg("_c0", "_c1").withColumnRenamed("avg(_c0)", "_c0").withColumnRenamed("avg(_c1)", "_c1")
    """Centers
    +-------+--------------------+-------------------+
    |cluster|                 _c0|                _c1|
    +-------+--------------------+-------------------+
    |    148|  0.9046112643478447| 0.7329637227031914|
    |     31|   0.761282446671807| 0.8231969928644237|
    | ...   | ...                | ...               |
    """
    
    # Convert to list so we can reduce them
    centerList = centers.collect()
    centerList = list(map(lambda x: {"clusters": [x["cluster"]], "points": [(x["_c0"], x["_c1"])]}, centerList))
    
    """Centerlist
    [{'cluster': 134, 'c0': 0.6814438096242257, 'c1': 0.9679323819232074}, {'cluster': 36, 'c0': 0.7768256711197831, 'c1': 0.04989300735420717},...]
    """
    # For each point, shove it in the closest cluster, O(n^2) but n is small (150)
    while len(centerList) > n:
        minDist = float("inf")
        minIndex = None
        minIndex2 = None
        for i in range(len(centerList)):
            for j in range(i+1, len(centerList)):
                for point1 in centerList[i]["points"]:
                    for point2 in centerList[j]["points"]:
                        d = dist(point1, point2)
                        if d < minDist:
                            minDist = d
                            minIndex = i
                            minIndex2 = j
        centerList[minIndex]["clusters"] += centerList[minIndex2]["clusters"]
        centerList[minIndex]["points"] += centerList[minIndex2]["points"]
        del centerList[minIndex2]
    
    # Map the clusters to the points
    clusterMap = {}
    for i, cluster in enumerate(centerList):
        for c in cluster["clusters"]:
            clusterMap[c] = i
    df = df.withColumn("realCluster", F.udf(lambda x: clusterMap[x], IntegerType())("cluster"))
    
    # Return the dataframe with the new clusters in column realCluster
    return df


# def plotDistribution(rdd: RDD):
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     import numpy as np
    
#     points = rdd.collect()
#     distances = [point[1][2] for point in points]
#     plt.hist(distances, bins=100)
#     plt.show()
    

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


if __name__ == "__main__":
    start_time = time.time()
    
    spark: SparkSession = SparkSession.builder.appName("dataminingProject").master("local[*]").getOrCreate()
    df: DF = spark.read.csv(sys.argv[1], header=False)
    
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