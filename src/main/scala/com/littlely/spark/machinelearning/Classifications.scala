package com.littlely.spark.machinelearning

import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel, VectorIndexer, VectorIndexerModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object Classifications {

  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("Classification")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    val classifications: Classifications = new Classifications(spark)
    classifications.getDecisionTreeClassifer()
  }
}


class Classifications(spark : SparkSession){

  // LogisticRegression model
  def getLogisticRegression(): Unit ={

    // 创建训练集
    val training: DataFrame = spark.createDataFrame(
      Seq(
        (1.0, Vectors.dense(0.0, 1.1, 0.1)),
        (2.0, Vectors.dense(2.0, 1.0, -1.0)),
        (0.0, Vectors.dense(2.0, 1.3, 1.0)),
        (1.0, Vectors.dense(0.0, 1.2, -0.5))
      )
    ).toDF("label", "features")
    // 创建regression实例
    val lr: LogisticRegression = new LogisticRegression()
    println("LogisticRegression params: \n" + lr.explainParams() + "\n")

    // 设置参数
    lr.setMaxIter(10).setRegParam(0.01)

    // 拟合模型
    val model: LogisticRegressionModel = lr.fit(training)
    println("Model was fit using params: " + model.parent.extractParamMap())

    // 也可使用ParamMap设定参数，后面值覆盖前面值
    val paramMap: ParamMap = ParamMap(lr.maxIter -> 20).put(lr.maxIter, 30).put(lr.regParam -> 0.1, lr.threshold -> 0.55)
    val paramMap1 = ParamMap(lr.probabilityCol -> "myProbability")  // 改变输出列名称
    val mapCombined: ParamMap = paramMap1 ++  paramMap  // 连接
    println("合并的参数： " + mapCombined)

    // 使用合并参数训练新模型
    val model_2: LogisticRegressionModel = lr.fit(training, mapCombined)
    println("Model_2 was fit using params: " + model_2.parent.extractParamMap())

    // 测试集
    val test: DataFrame = spark.createDataFrame(
      Seq(
        (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
        (0.0, Vectors.dense(3.0, 2.0, -0.1)),
        (1.0, Vectors.dense(0.0, 2.2, -1.5))
      )
    ).toDF("label", "features")

    model_2.transform(test).select("features", "label", "myProbability", "prediction").collect()
      .foreach{case Row(features : Vector, label : Double, prob : Vector, prediction : Double) =>
        println(s"($features, $label) -> prob=$prob, prediction=$prediction")}
  }

  def getDecisionTreeClassifer(): Unit ={
    // load data from hdfs
    val data: DataFrame = spark.read.format("libsvm").load("hdfs://centos03:9000/spark-data/mllib/sample_libsvm_data.txt")

    val labelIndexer: StringIndexerModel = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
    val featureIndexer: VectorIndexerModel = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
    labelIndexer.transform(data).show(3)
    featureIndexer.transform(data).show(3)

    val Array(trainingData, testData): Array[Dataset[Row]] = data.randomSplit(Array(0.7, 0.3))

    val dt: DecisionTreeClassifier = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

    // Convert indexed labels back to original labels
    val labelConverter: IndexToString = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    val pipeline: Pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    //train model
    val model: PipelineModel = pipeline.fit(trainingData)
    val predictions: DataFrame = model.transform(testData)

    predictions.show(5) //select("predictedLabel", "label", "features").show(5)

    // calc test error
    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val acc: Double = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - acc))

    // model.stages(2) means the second stage of model: dt
    val treeModel: DecisionTreeClassificationModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model: " + treeModel.toDebugString)

  }
}
