package com.littlely.spark.machinelearning

import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector

import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object Regressions {

  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("Regression")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    //    val testReg: TestRegression = new TestRegression(spark)
    //    testReg.getLogisticRegression()

    //测试pipeline
    val testPipeline: TestPipeline = new TestPipeline(spark)
    testPipeline.getPipeline()

  }

}

class TestRegression(spark : SparkSession){


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
