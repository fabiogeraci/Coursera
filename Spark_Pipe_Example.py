import warnings
warnings.filterwarnings(action='ignore', category=Warning)

# Apache Spark run on Ubuntu via Pycharm
import findspark
import requests
import os.path
findspark.init()

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-1.8.0-openjdk-amd64"
os.environ["SPARK_HOME"] = "/opt/spark"

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
except ImportError as e:
    print('<<<<<!!!!! Please restart your kernel after installing Apache Spark !!!!!>>>>>')


spark = SparkSession \
    .builder \
    .appName("Python Spark SQL") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
#
save_path = os.path.join(os.getcwd(), 'data/hmp.parquet')
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
    options(header='true',inferschema='true').load("data/hmp.parquet",header=True)

from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer, MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline

indexer = StringIndexer(inputCol="class", outputCol="classIndex")
encoder = OneHotEncoder(inputCol="classIndex", outputCol="categoryVec")
vectorAssembler = VectorAssembler(inputCols=["x","y","z"],
                                  outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

#minmaxscaler = MinMaxScaler(inputCol="features", outputCol="features_norm",min=0,max=1)

pipeline = Pipeline(stages=[indexer, encoder, vectorAssembler, normalizer])
#pipeline = Pipeline(stages=[indexer, encoder, vectorAssembler, minmaxscaler])
model = pipeline.fit(df)
prediction = model.transform(df)
#print(prediction.show(10))


# kmeans pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

kms = 10
silhouettes = []

for k_set in range(2, kms):

    kmeans = KMeans(featuresCol="features").setK(k_set).setSeed(1)
    pipeline = Pipeline(stages=[vectorAssembler, kmeans])
    model = pipeline.fit(df)
    predictions = model.transform(df)

    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    silhouettes.append(silhouette)

print(silhouettes)
print("The best accuracy was with {:.3f}".format(max(silhouettes)), "with k=",
      silhouettes.index(max(silhouettes))+2)
# print("Silhouette with squared euclidean distance = " + str(silhouette))

kmeans = KMeans(featuresCol="features").setK(5).setSeed(1)
pipeline = Pipeline(stages=[vectorAssembler, kmeans, normalizer])
model = pipeline.fit(df)

predictions = model.transform(df)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

import pyspark.sql.functions as F
df_denormalized = df.select([F.col('*'),(F.col('x')*10)]).drop('x').withColumnRenamed('(x * 10)','x')

kmeans = KMeans(featuresCol="features").setK(14).setSeed(1)
pipeline = Pipeline(stages=[vectorAssembler, kmeans])
model = pipeline.fit(df_denormalized)
predictions = model.transform(df_denormalized)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

from pyspark.ml.clustering import GaussianMixture

gmm = GaussianMixture(featuresCol="features").setK(2).setSeed(1)
pipeline = pipeline = Pipeline(stages=[vectorAssembler, gmm])

model = pipeline.fit(df)

predictions = model.transform(df)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("GaussianMixture = " + str(silhouette))