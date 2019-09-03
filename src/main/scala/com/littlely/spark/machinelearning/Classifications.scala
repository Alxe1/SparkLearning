package com.littlely.spark.machinelearning

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object Classifications {

  def main(args: Array[String]): Unit = {

  }
}


class Classifications(spark : SparkSession){

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
}
