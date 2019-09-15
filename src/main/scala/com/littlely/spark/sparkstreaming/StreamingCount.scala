package com.littlely.spark.sparkstreaming

import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.SparkConf
import org.apache.spark.streaming._
import org.apache.spark.streaming.dstream.{DStream, InputDStream}
import org.apache.spark.streaming.kafka010.{ConsumerStrategies, KafkaUtils, LocationStrategies}


object StreamingCount {

  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("SparkStreaming")
    val ssc = new StreamingContext(conf, Seconds(5))

    val kafkaParam: Map[String, Object] = Map(
      "bootstrap.servers" -> "centos03:9092",
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> "llgroup",
      "auto.offset.reset" -> "latest"
    )
    val kafkaDStream: InputDStream[ConsumerRecord[String, String]] = KafkaUtils.createDirectStream[String, String](ssc, LocationStrategies.PreferConsistent,
      ConsumerStrategies.Subscribe[String, String](Array("ll"), kafkaParam))

    val wordsMap: DStream[(String, Int)] = kafkaDStream.flatMap(record => record.value().split(" ")).map((_, 1)).groupByKey().map(t => (t._1, t._2.sum))
    wordsMap.print()

    ssc.start()
    ssc.awaitTermination()

  }
}



