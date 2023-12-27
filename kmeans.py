import pyspark.sql.dataframe as DF
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def kmeansCluster(df: DF, n: int):
    vec_assembler = VectorAssembler(inputCols = ["_c0", "_c1"], 
                                    outputCol='features', handleInvalid="skip") 
    
    final_data = vec_assembler.transform(df)

    silhouette_score=[] 
  
    kmeans = KMeans(k=n, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(final_data)
    clustered = model.transform(final_data)
    
    return clustered.select("_c0", "_c1", "cluster")

def reduceClusters(df: DF, n: int):
    """in
    +--------------------+-------------------+-------+                              
    |                 _c0|                _c1|cluster|
    +--------------------+-------------------+-------+
    |  0.8095605242868158| 0.9369266055045872|    110|
    | 0.09348496530454897|  0.944954128440367|     50|
                ...             ...             ... 
    """
    
    return df.groupBy("cluster").avg("_c0", "_c1").withColumnRenamed("avg(_c0)", "_c0").withColumnRenamed("avg(_c1)", "_c1")
    
    
    # return clustered.select("_c0", "_c1", "cluster")