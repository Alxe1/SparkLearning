package com.littlely.spark.machinelearning

import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer, VectorIndexer, VectorIndexerModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor, GeneralizedLinearRegression, GeneralizedLinearRegressionModel, GeneralizedLinearRegressionTrainingSummary, LinearRegression, LinearRegressionModel, LinearRegressionTrainingSummary, RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object Regressions {

  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("Regression")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    val regressions = new Regressions(spark)
    regressions.getRandomForestRegression()

  }

}

class Regressions(spark : SparkSession){

  def getLinearRegression(): Unit ={

    val training: DataFrame = spark.read.format("libsvm").load("hdfs://centos03:9000/spark-data/mllib/sample_linear_regression_data.txt")
    training.show(5, false)

    val lr: LinearRegression = new LinearRegression().setMaxIter(20).setRegParam(0.3).setElasticNetParam(0.8)
    val lrModel: LinearRegressionModel = lr.fit(training)
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model
    val trainingSummary: LinearRegressionTrainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString("," )}]")
    trainingSummary.residuals.show()
    println(s"p Value: ${trainingSummary.pValues}")
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  }

  def getGeneralizedLinearRegression(): Unit ={

    val training: DataFrame = spark.read.format("libsvm").load("hdfs://centos03:9000/spark-data/mllib/sample_linear_regression_data.txt")
    training.show(5, false)

    val glr: GeneralizedLinearRegression = new GeneralizedLinearRegression().setFamily("gaussian").setLink("identity").setMaxIter(10).setRegParam(0.3)
    val model: GeneralizedLinearRegressionModel = glr.fit(training)
    println(s"Coefficients: ${model.coefficients}")
    println(s"Intercept: ${model.intercept}")

    val summary: GeneralizedLinearRegressionTrainingSummary = model.summary

    println(s"Coefficient Standard Error: ${summary.coefficientStandardErrors.mkString(",")}")
    println(s"T values: ${summary.tValues.mkString(",")}")
    println(s"P values: ${summary.pValues.mkString(",")}")
    println(s"Dispersion: ${summary.dispersion}")
    println(s"Null Deviance: ${summary.nullDeviance}")
    println(s"Residual Degree Of Freedom Null: ${summary.residualDegreeOfFreedomNull}")
    println(s"Deviance: ${summary.deviance}")
    println(s"Residual Degree Of Freedom: ${summary.residualDegreeOfFreedom}")
    println(s"AIC: ${summary.aic}")
    println("Deviance Residuals: ")
    summary.residuals().show()
  }

  def getDecisionTreeRegression(): Unit ={
    val data: DataFrame = spark.read.format("libsvm").load("hdfs://centos03:9000/spark-data/mllib/sample_libsvm_data.txt")
    data.show(5, false)

    val featureIndexer: VectorIndexerModel = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
    val Array(trainingData, testData): Array[Dataset[Row]] = data.randomSplit(Array(0.7, 0.3))

    val dt: DecisionTreeRegressor = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")

    val pipeline: Pipeline = new Pipeline().setStages(Array(featureIndexer, dt))
    val model: PipelineModel = pipeline.fit(trainingData)

    val predictions: DataFrame = model.transform(testData)
    predictions.show(5, false)

    val evaluator: RegressionEvaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    val rmse: Double = evaluator.evaluate(predictions)
    println("Root Mean Squared Error(RMSE) on test data = " + rmse)

    val treeModel: DecisionTreeRegressionModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model: \n" + treeModel.toDebugString)
  }

  def getRandomForestRegression(): Unit ={
    val data: DataFrame = spark.read.format("libsvm").load("hdfs://centos03:9000/spark-data/mllib/sample_libsvm_data.txt")
    data.show(5, false)

    val featureIndexer: VectorIndexerModel = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
    val Array(trainingData, testData): Array[Dataset[Row]] = data.randomSplit(Array(0.7, 0.3))

    val rf: RandomForestRegressor = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")

    val pipeline: Pipeline = new Pipeline().setStages(Array(featureIndexer, rf))
    val model: PipelineModel = pipeline.fit(trainingData)

    val predictions: DataFrame = model.transform(testData)
    predictions.show(5, false)

    val evaluator: RegressionEvaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    val rmse: Double = evaluator.evaluate(predictions)
    println("RMSE ON TEST DATA: " + rmse)

    val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
    println("Learned regression forest model:\n" + rfModel.toDebugString)
  }
}


class TestPipeline(spark : SparkSession){

  def getPipeline(): Unit ={

    val training: DataFrame = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "text", "label")

    val tokenizer: Tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val hashingTF: HashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
    val lr: LogisticRegression = new LogisticRegression().setMaxIter(10).setRegParam(0.001)
    val pipeline: Pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

    // 拟合数据
    val model: PipelineModel = pipeline.fit(training)

    // 保存模型及没有拟合数据的pipeline
    // TODO: 注意在菜单栏Run的edit configurations中设置VM options参数-DHADOOP_USER_HOME=root -Dspark.master=spark://centos03:8080
    model.write.overwrite().save("hdfs://centos03:9000/logistic/spark-logistic-regression-model")
    pipeline.write.overwrite().save("hdfs://centos03:9000/logistic/unfit-lr-model")

    // 加载模型
    val sameModel: PipelineModel = PipelineModel.load("hdfs://centos03:9000/logistic/spark-logistic-regression-model")

    val test: DataFrame = spark.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "spark hadoop spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")


    model.transform(test).select("id", "text", "probability", "prediction").collect()
      .foreach{ case Row(id : Long, text : String, prob : Vector, prediction : Double) =>
          println(s"model: ($id, $text) --> prob=$prob, prediction=$prediction")
      }

    sameModel.transform(test).select("id", "text", "probability", "prediction").collect()
      .foreach{ case Row(id : Long, text : String, prob : Vector, prediction : Double) =>
        println(s"sameModel: ($id, $text) --> prob=$prob, prediction=$prediction")
      }

  }
}
