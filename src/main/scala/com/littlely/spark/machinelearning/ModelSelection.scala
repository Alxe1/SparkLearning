package com.littlely.spark.machinelearning

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.LinearRegression

object ModelSelection {

  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("ModelSelection")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    val modelSelection = new ModelSelection(spark)
    modelSelection.getCrossValidator()

  }
}

class ModelSelection(spark: SparkSession){

  def getCrossValidator(): Unit ={
    val training: DataFrame = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0),
      (4L, "b spark who", 1.0),
      (5L, "g d a y", 0.0),
      (6L, "spark fly", 1.0),
      (7L, "was mapreduce", 0.0),
      (8L, "e spark program", 1.0),
      (9L, "a e c l", 0.0),
      (10L, "spark compile", 1.0),
      (11L, "hadoop software", 0.0)
    )).toDF("id", "text", "label")

    val test = spark.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "mapreduce spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")

    val tokenizer: Tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val hashingTF: HashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("features")
    val lr: LogisticRegression = new LogisticRegression().setMaxIter(10)
    val pipeline: Pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

    val paramGrid: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.01)).build()

    val cv: CrossValidator = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator().setMetricName("areaUnderROC"))
      .setEstimatorParamMaps(paramGrid).setNumFolds(2)
    val cvModel: CrossValidatorModel = cv.fit(training)
    println(cv.explainParams())
    val predictions: DataFrame = cvModel.transform(test)
    predictions.show(5, false)

    predictions.select("id", "text", "probability", "prediction").collect()
      .foreach{case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
      println(s"($id, $text) --> prob=$prob, prediction=$prediction")
      }
  }

  def getTrainValidationSplit(): Unit ={

    val data: DataFrame = spark.read.format("libsvm").load("hdfs://centos03:9000/spark-data/mllib/sample_linear_regression_data.txt")
    data.show(5, false)
    val Array(training, test): Array[Dataset[Row]] = data.randomSplit(Array(0.9, 0.1), seed=12345L)

    val lr: LinearRegression = new LinearRegression().setMaxIter(10)
    val paramGrid: Array[ParamMap] = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept).addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1)).build()

    val trainValidationSplit: TrainValidationSplit = new TrainValidationSplit().setEstimator(lr)
      .setEvaluator(new RegressionEvaluator().setMetricName("rmse"))
      .setEstimatorParamMaps(paramGrid).setTrainRatio(0.8)  // 90%的训练数据中，80%用于训练， 20%用于验证

    val model: TrainValidationSplitModel = trainValidationSplit.fit(training)

    model.transform(test).select("features", "label", "prediction").show()

  }
}