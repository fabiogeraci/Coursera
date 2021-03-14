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


from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")


print(data.show(5))
#
# # Index labels, adding metadata to the label column.
# # Fit on whole dataset to include all labels in index.
#
#
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)


#
# # Automatically identify categorical features, and index them.
# # Set maxCategories so features with > 4 distinct values are treated as continuous.
# featureIndexer =\
#     VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
#
# # Split the data into training and test sets (30% held out for testing)
# (trainingData, testData) = data.randomSplit([0.7, 0.3])
#
# # Train a RandomForest model.
# rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
#
# # Convert indexed labels back to original labels.
# labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
#                                labels=labelIndexer.labels)
#
# # Chain indexers and forest in a Pipeline
# pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])
#
# # Train model.  This also runs the indexers.
# model = pipeline.fit(trainingData)
#
# # Make predictions.
# predictions = model.transform(testData)
#
# # Select example rows to display.
# predictions.select("predictedLabel", "label", "features").show(5)
#
# # Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
# accuracy = evaluator.evaluate(predictions)
# print("Test Error = %g" % (1.0 - accuracy))
#
# rfModel = model.stages[2]
# print(rfModel)  # summary only