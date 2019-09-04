package com.littlely.spark.machinelearning

import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.{GaussianMixture, GaussianMixtureModel, KMeans, KMeansModel, LDA, LDAModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Clusters {

  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("Clusters")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    val clusters = new Clusters(spark)
    clusters.getGMM()
  }
}

class Clusters(spark : SparkSession){

  def getKMeans(): Unit ={
    val dataset: DataFrame = spark.read.format("libsvm").load("hdfs://centos03:9000/spark-data/mllib/sample_kmeans_data.txt")
    dataset.show(5, false)

    val kmeans: KMeans = new KMeans().setK(2).setSeed(1L)
    val model: KMeansModel = kmeans.fit(dataset)

    //计算组内误差平方和
    val WSSSE: Double = model.computeCost(dataset)
    println(s"Within Set Sum of Squared Errors = $WSSSE")
    println("Cluster center: \n")
    model.clusterCenters.foreach(println)
  }

  def getLDA(): Unit ={
    val dataset: DataFrame = spark.read.format("libsvm").load("hdfs://centos03:9000/spark-data/mllib/sample_lda_libsvm_data.txt")
    dataset.show(5, false)

    val lda: LDA = new LDA().setK(10).setMaxIter(20)
    val model: LDAModel = lda.fit(dataset)

    val ll: Double = model.logLikelihood(dataset)
    val lp: Double = model.logPerplexity(dataset)
    println(s"The lower bound on the log likelihood of the entire corpus: $ll")
    println(s"The upper bound on perplexity: $lp")

    // 描述主题
    val topics: DataFrame = model.describeTopics(4)
    println("The topics described by their top-weighted terms:")
    topics.show(10, false)

    val transformed: DataFrame = model.transform(dataset)
    transformed.show(false)
  }

  def getGMM(): Unit ={
    val dataset: DataFrame = spark.read.format("libsvm").load("hdfs://centos03:9000/spark-data/mllib/sample_kmeans_data.txt")
    dataset.show(5, false)

    val gmm: GaussianMixture = new GaussianMixture().setK(2)
    val model: GaussianMixtureModel = gmm.fit(dataset)

    for (i <- 0 until model.getK){
      println(s"Gaussian $i \nweight=${model.weights(i)}\nmu=${model.gaussians(i).mean}\nsigma=\n${model.gaussians(i).cov}")
    }
  }

}

