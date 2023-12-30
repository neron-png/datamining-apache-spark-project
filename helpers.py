from pyspark.sql.dataframe import DataFrame as DF
from pyspark.sql import functions as F


def cleanup(df: DF) -> DF:
    cleandf = df.dropna(how='any')
    return cleandf

def normalize(df: DF) -> DF:
    min0 = float(df.agg(F.min("_c0")).collect()[0][0])
    max0 = float(df.agg(F.max("_c0")).collect()[0][0])
    min1 = float(df.agg(F.min("_c1")).collect()[0][0])
    max1 = float(df.agg(F.max("_c1")).collect()[0][0])

    df_normalized = df.withColumn("_c0", (df["_c0"] - min0) / (max0 - min0))\
        .withColumn("_c1", (df["_c1"] - min1) / (max1 - min1))
    
    return df_normalized

def plot(df: DF, featureColumn: str):
    import matplotlib.pyplot as plt
    import pandas as pd

    pdf = df.toPandas()
    pdf.plot.scatter(x='_c0', y='_c1', c=featureColumn, cmap="viridis")
    plt.show()

def testing_listPlot(df: list):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    plt.scatter(x=[_[0] for _ in df], y=[_[1] for _ in df], c=[_[2] for _ in df], cmap="viridis")
    plt.show()