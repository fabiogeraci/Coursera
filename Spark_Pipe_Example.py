# Apache Spark run on Ubuntu via Pycharm
import requests
import os.path


try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
except ImportError as e:
    print('<<<<<!!!!! Please restart your kernel after installing Apache Spark !!!!!>>>>>')

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


#
save_path = os.path.join(os.getcwd(), 'hmp.parquet')
print(save_path)
url = 'https://github.com/IBM/coursera/raw/master/hmp.parquet'


def download_url(url_, save_path_, chunk_size=128):
    r = requests.get(url_, stream=True)
    with open(save_path_, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


if not os.path.isfile(save_path):
    download_url(url, save_path)
    print('End Downloading')
else:
    print('File Already Exists')

df = spark.read.format('parquet').\
    options(header='true',inferschema='true').load("hmp.parquet",header=True)

from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer, MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline

indexer = StringIndexer(inputCol="class", outputCol="classIndex")
encoder = OneHotEncoder(inputCol="classIndex", outputCol="categoryVec")
vectorAssembler = VectorAssembler(inputCols=["x","y","z"],
                                  outputCol="features")
#normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

minmaxscaler = MinMaxScaler(inputCol="features", outputCol="features_norm",min=0,max=1)

pipeline = Pipeline(stages=[indexer, encoder, vectorAssembler, minmaxscaler])
model = pipeline.fit(df)
prediction = model.transform(df)
print(prediction.show(10))