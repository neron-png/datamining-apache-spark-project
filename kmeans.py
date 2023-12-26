import pyspark.sql.dataframe as DF
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def kmeansCluster(df: DF, n: int):
    vec_assembler = VectorAssembler(inputCols = ["_c0", "_c1"], 
                                    outputCol='features', handleInvalid="skip") 
    
    final_data = vec_assembler.transform(df)

    silhouette_score=[] 
  
    kmeans = KMeans(k=5, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(final_data)
    clustered = model.transform(final_data)
    
    return clustered.select("_c0", "_c1", "cluster")