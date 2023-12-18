from pyspark.sql import DataFrame as DF
from pyspark import RDD

def cleanup(df: DF) -> DF:
    
    cleandf = df.dropna()
    return cleandf

def normalize(df: DF) -> DF:
    pass
    return normalized_df