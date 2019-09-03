package com.littlely.spark.datatransform

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.SparkConf


case class User(id : Int, name : String, age : Int)

object RDDTransform {

  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("RDDTransform")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    // RDD与DataFrame或DataSet之间转换前需要引入隐式转换
    import spark.implicits._

    // 创建RDD
    val rdd: RDD[(Int, String, Int)] = spark.sparkContext.makeRDD(List((1, "zhangs", 20), (2, "lis", 30), (3, "wangw", 40)))
    // RDD转换为DF(需要有结构)
    val df: DataFrame = rdd.toDF("id", "name", "age")
    // DF转换为DS(需要有类型)
    val ds: Dataset[User] = df.as[User]
    // DS转换为DF
    val df1: DataFrame = ds.toDF()
    // DF转换为RDD
    val rdd1: RDD[Row] = df1.rdd
    // DS转换为RDD
    val rdd2: RDD[User] = ds.rdd

    println("rdd1:\n")
    rdd1.foreach(row => {println(row.getString(1))})
    println("rdd2:\n")
    rdd2.foreach(row => {println(row.name)})

    spark.stop()
  }

}
