package com.littlely.spark.baseops

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD


object BaseOps {

  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("BaseOps")
    val sc: SparkContext = new SparkContext(conf)
    // 读取文件
    val text: RDD[String] = sc.textFile("file:///F:\\java_program\\Sparks\\src\\main\\scala\\sample\\hello.txt") // hdfs://centos03:9000/sample/hello.txt
    val rdd: RDD[(String, Int)] = sc.parallelize(List(("a", 3), ("a", 2), ("c", 4), ("b", 3), ("c", 6), ("c", 8)), 2)

    // word count
//    val wordCount: WordCount = new WordCount
//    wordCount.wordCount(text)

    val maxAdd: MaxValueAdd = new MaxValueAdd
    maxAdd.getMaxValueAdd(rdd)

  }

}

// 单词计数
class WordCount{

  def wordCount(text : RDD[String]) : Unit = {

    val flatRDD: RDD[String] = text.flatMap(_.split(" "))
    val mapRDD: RDD[(String, Int)] = flatRDD.map((_, 1))
    val reduceRDD: RDD[(String, Int)] = mapRDD.reduceByKey(_+_)
    val result: RDD[(String, Int)] = reduceRDD.sortByKey(false)
    result.foreach(println)
    // 可以一行代码
//    text.flatMap(_.split(" ")).map((_, 1)).reduceByKey(_+_).sortByKey(false).foreach(println)
  }
}

class WordCount1{

  def wordCount(text : RDD[String]) : Unit = {

    val flatRDD: RDD[String] = text.flatMap(_.split(" "))
    val mapRDD: RDD[(String, Int)] = flatRDD.mapPartitions(datas=>{datas.map(data=>(data, 1))})
    val groupedRDD: RDD[(String, Iterable[Int])] = mapRDD.groupByKey()
    val result: RDD[(String, Int)] = groupedRDD.map(t => (t._1, t._2.sum))
    result.foreach(println)
  }
}

class MaxValueAdd{

  def getMaxValueAdd(rdd : RDD[(String, Int)]) : Unit = {

    // 求分区内相同键的最大值，并对不同分区之间的最大值进行相加
    rdd.glom().foreach(x=>{x.foreach(println)})
    val result: RDD[(String, Int)] = rdd.aggregateByKey(0)(Math.max(_, _), _+_)
    result.foreach(println)

    // 统计每个字母的总值及其出现次数
    val result1: Array[(String, (Int, Int))] = rdd.combineByKey(
      (_, 1),
      (acc: (Int, Int), v) => (acc._1 + v, acc._2 + 1),
      (acc1: (Int, Int), acc2: (Int, Int)) => (acc1._1 + acc2._1, acc1._2 + acc2._2)
    ).collect()
    println(result1.mkString("||"))
  }
}


