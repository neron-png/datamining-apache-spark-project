from pyspark.sql.dataframe import DataFrame as DF
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.types import IntegerType

def kmeansCluster(df: DF, n: int):
    vec_assembler = VectorAssembler(inputCols = ["_c0", "_c1"], 
                                    outputCol='features', handleInvalid="skip") 
    
    final_data = vec_assembler.transform(df)

    silhouette_score=[] 
  
    kmeans = KMeans(k=n, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(final_data)
    clustered = model.transform(final_data)
    
    return clustered.select("_c0", "_c1", "cluster")


def dist(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def reduceClusters(df: DF, n: int):
    from pyspark.ml.clustering import BisectingKMeans
    """in
    +--------------------+-------------------+-------+                              
    |                 _c0|                _c1|cluster|
    +--------------------+-------------------+-------+
    |  0.8095605242868158| 0.9369266055045872|    110|
    | 0.09348496530454897|  0.944954128440367|     50|
                ...             ...             ... 
    """
    # rdd = df.rdd.map(lambda x: (x[2], (x[0], x[1])))
    # rdd = rdd.groupByKey().map(lambda x: (x[0], list(x[1])))
    # rdd = rdd.reduceByKey(lambda x, y: x + y)
    
    # Group by cluster and vectorize it's points
    centers = df.groupBy("cluster").avg("_c0", "_c1").withColumnRenamed("avg(_c0)", "_c0").withColumnRenamed("avg(_c1)", "_c1")
    vec_assembler = VectorAssembler(inputCols = ["_c0", "_c1"], 
                                    outputCol='features', handleInvalid="skip") 
    transformed_centers = vec_assembler.transform(centers)
    
    bkm = BisectingKMeans(featuresCol="features", predictionCol="smallCluster", k=n, seed=42)
    model = bkm.fit(transformed_centers)
    predictions = model.transform(transformed_centers)
    predictions.show()
    
    
    return predictions.select("_c0", "_c1", "smallCluster").withColumnRenamed("smallCluster", "cluster")
    



def single_link(df: DF, n: int):
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