package com.littlely.spark.userdefinedfunction

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Encoders, Row, SparkSession, TypedColumn}
import org.apache.spark.sql.expressions.{Aggregator, MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{DataType, DoubleType, LongType, StructType}


case class UserBean(name : String, age : BigInt)
case class AvgBuffer(var sum : BigInt, var count : Int)


object UDFFunction {

  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("udfFunc")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    //转换前需要引入隐式转换
    import spark.implicits._

    // 获取数据
    val df: DataFrame = spark.read.json("file:///F:\\java_program\\Sparks\\src\\main\\scala\\sample\\users.json")
//    val df: DataFrame = spark.read.json("hdfs://centos03:9000/datas/users.json")

    // 基本sql操作
    df.createOrReplaceTempView("user")
    spark.sql("select * from user").show()

    // 使用自定义聚合函数，创建弱类型聚合函数对象
    val udfAvg: MyAvgFunction = new MyAvgFunction
    // 注册
    spark.udf.register("udfAvg", udfAvg)
    df.createOrReplaceTempView("student")
    spark.sql("select udfAvg(age) from student").show()

    // 使用自定义聚合函数，创建强类型聚合函数对象
    val udfAvg2: MyAvgFunction2 = new MyAvgFunction2
    // 将聚合函数转换为查询列
    val avgCol: TypedColumn[UserBean, Double] = udfAvg2.toColumn.name("udfAvg2")
    val userDS: Dataset[UserBean] = df.as[UserBean]
    userDS.select(avgCol).show()


    spark.stop()

  }

}


// 自定义聚合函数（弱类型）
class MyAvgFunction extends UserDefinedAggregateFunction{

  // 函数输入的数据结构
  override def inputSchema: StructType = {
    new StructType().add("age", LongType)
  }

  // 计算时的数据结构
  override def bufferSchema: StructType = {
    new StructType().add("sum", LongType).add("count", LongType)
  }

  // 函数返回的数据类型
  override def dataType: DataType = DoubleType

  // 函数是否稳定
  override def deterministic: Boolean = true

  // 计算前缓冲区的初始化
  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = 0L
    buffer(1) = 0L
  }

  // 根据查询结果更新缓冲区数据
  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    buffer(0) = buffer.getLong(0) + input.getLong(0)
    buffer(1) = buffer.getLong(1) + 1
  }

  // 将多个节点的缓冲区合并
  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1(0) = buffer1.getLong(0) + buffer2.getLong(0)
    buffer1(1) = buffer1.getLong(1) + buffer2.getLong(1)
  }

  // 计算
  override def evaluate(buffer: Row): Any = {
    buffer.getLong(0).toDouble / buffer.getLong(1)
  }
}


// 自定义聚合函数（强类型）
class MyAvgFunction2 extends Aggregator[UserBean, AvgBuffer, Double]{
  // 初始化
  override def zero: AvgBuffer = {
    AvgBuffer(0, 0)
  }

  // 数据聚合
  override def reduce(b: AvgBuffer, a: UserBean): AvgBuffer = {
    b.sum = b.sum + a.age
    b.count = b.count + 1
    b  // return b
  }

  // 缓冲区合并
  override def merge(b1: AvgBuffer, b2: AvgBuffer): AvgBuffer = {
    b1.sum = b1.sum + b2.sum
    b1.count = b1.count + b2.count
    b1
  }

  // 完成计算
  override def finish(reduction: AvgBuffer): Double = {
    reduction.sum.toDouble / reduction.count
  }

  override def bufferEncoder: Encoder[AvgBuffer] = Encoders.product

  override def outputEncoder: Encoder[Double] = Encoders.scalaDouble
}
