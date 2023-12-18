from helpers import cleanup, normalize
import pyspark
from pyspark import SparkContext as sc

if __name__ == "__main__":
    
    rdd = sc.textFile("data-example2224.csv")
    
    # TODO: implement
    clean_rdd = cleanup(rdd)
    normalized_rdd = normalize(clean_rdd)
    
    print("Hello World!")