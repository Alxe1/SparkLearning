package com.littlely.spark.machinelearning

import java.util

import org.apache.spark.SparkConf
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.feature.{Binarizer, BucketedRandomProjectionLSH, BucketedRandomProjectionLSHModel, ChiSqSelector, CountVectorizer, CountVectorizerModel, HashingTF, IDF, IDFModel, NGram, Normalizer, PCA, PCAModel, RegexTokenizer, StopWordsRemover, Tokenizer, VectorSlicer, Word2Vec, Word2VecModel}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

object FeatureEngineer {

  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("FeatureEngineer")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    val featureSelectors = new FeatureSelectors(spark)
    featureSelectors.getLSH()

  }
}

class FeatureSelectors(spark : SparkSession){

  def getVectorSlicer(): Unit ={
    val data: util.List[Row] = util.Arrays.asList(
      Row(Vectors.sparse(3, Seq((0, -2.0), (1, 2.3)))),
      Row(Vectors.dense(-2.0, 2.3, 0.0))
    )

    val defaultAttr: NumericAttribute = NumericAttribute.defaultAttr
    val attrs: Array[NumericAttribute] = Array("f1", "f2", "f3").map(defaultAttr.withName)
    val attrGroup = new AttributeGroup("userFeatures", attrs.asInstanceOf[Array[Attribute]])

    val dataset: DataFrame = spark.createDataFrame(data, StructType(Array(attrGroup.toStructField())))
    val slicer: VectorSlicer = new VectorSlicer().setInputCol("userFeatures").setOutputCol("features")
    slicer.setIndices(Array(1)).setNames(Array("f3"))

    val output: DataFrame = slicer.transform(dataset)
    output.show()
  }

  def getChiSqSelector(): Unit ={

    val data = Seq(
      (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
      (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
      (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
    )

    val df: DataFrame = spark.createDataFrame(data).toDF("id", "features", "clicked")

    val selector: ChiSqSelector = new ChiSqSelector().setNumTopFeatures(2).setFeaturesCol("features").setLabelCol("clicked").setOutputCol("selected")

    val result: DataFrame = selector.fit(df).transform(df)
    println(s"ChiSqSlector output with top ${selector.getNumTopFeatures} features selected")
    result.show(false)
  }

  def getLSH(): Unit ={

    val dfA = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0)),
      (1, Vectors.dense(1.0, -1.0)),
      (2, Vectors.dense(-1.0, -1.0)),
      (3, Vectors.dense(-1.0, 1.0))
    )).toDF("id", "keys")

    val dfB = spark.createDataFrame(Seq(
      (4, Vectors.dense(1.0, 0.0)),
      (5, Vectors.dense(-1.0, 0.0)),
      (6, Vectors.dense(0.0, 1.0)),
      (7, Vectors.dense(0.0, -1.0))
    )).toDF("id", "keys")

    val key: Vector = Vectors.dense(1.0, 0.0)
    val brp: BucketedRandomProjectionLSH = new BucketedRandomProjectionLSH().setBucketLength(2.0).setNumHashTables(3).setInputCol("keys").setOutputCol("values")

    val model: BucketedRandomProjectionLSHModel = brp.fit(dfA)
    model.transform(dfA).show()

    // 缓存转换的列
    val transformedA: DataFrame = model.transform(dfA).cache()
    val transformedB: DataFrame = model.transform(dfB).cache()

    // similarity join
    model.approxSimilarityJoin(dfA, dfB, 1.5).show()
    model.approxSimilarityJoin(transformedA, transformedB, 1.5).show()
    // self join
    model.approxSimilarityJoin(dfA, dfA, 2.5).filter("datasetA.id < datasetB.id").show()

    // 相似最近邻搜索
    model.approxNearestNeighbors(dfA, key, 2).show()
    model.approxNearestNeighbors(transformedA, key, 2).show()

  }
}

class FeatureTransformers(spark : SparkSession){

  def getNormalizer(): Unit ={
    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.5, -1.0)),
      (1, Vectors.dense(2.0, 1.0, 1.0)),
      (2, Vectors.dense(4.0, 10.0, 2.0))
    )).toDF("id", "features")

    val normalizer: Normalizer = new Normalizer().setInputCol("features").setOutputCol("normFeatures").setP(1.0)
    val l1NormData: DataFrame = normalizer.transform(dataFrame)
    println("Normalized using L^1 norm")
    l1NormData.show()

    val lInfNormData: DataFrame = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity)
    println("Normalized using L^inf norm")
    lInfNormData.show()
  }

  def getTokens(): Unit ={
    val sentenceDF: DataFrame = spark.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (1, "I wish Java could use case classes"),
      (2, "Logistic,regression,models,are,neat")
    )).toDF("id", "sentence")

    val tokenizer: Tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    new RegexTokenizer().setInputCol("sentence").setOutputCol("words").setPattern("\\W")  // .setGaps(false)

    val countTokens: UserDefinedFunction = udf{words: Seq[String] => words.length}
    val tokenized: DataFrame = tokenizer.transform(sentenceDF)
    tokenized.select("sentence", "words").withColumn("tokens", countTokens(col("words"))).show(false)
  }

  def getTokenRemover(): Unit ={
    val remover: StopWordsRemover = new StopWordsRemover().setInputCol("raw").setOutputCol("filtered")
    val dataSet: DataFrame = spark.createDataFrame(Seq(
      (0, Seq("I", "saw", "the", "red", "balloon")),
      (1, Seq("Mary", "had", "a", "little", "lamb"))
    )).toDF("id", "raw")

    remover.transform(dataSet).show(false)
  }

  def getNGram(): Unit ={
    val wordDataFrame = spark.createDataFrame(Seq(
      (0, Array("Hi", "I", "heard", "about", "Spark")),
      (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
      (2, Array("Logistic", "regression", "models", "are", "neat"))
    )).toDF("id", "words")

    val ngram: NGram = new NGram().setN(2).setInputCol("words").setOutputCol("ngrams")
    val ngramDF: DataFrame = ngram.transform(wordDataFrame)
    ngramDF.select("ngrams").show(false)
  }

  def getBinarizer(): Unit ={
    val data: Array[(Int, Double)] = Array((0, 0.1), (1, 0.8), (2, 0.2))
    val df: DataFrame = spark.createDataFrame(data).toDF("id", "feature")

    val binarizer: Binarizer = new Binarizer().setInputCol("feature").setOutputCol("bi_feature").setThreshold(0.5)
    val biDataFrame: DataFrame = binarizer.transform(df)

    println(s"Binarizer output with Threshold = ${binarizer.getThreshold}")
    biDataFrame.show(false)

  }

  def getPCA(): Unit ={
    val data = Array(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )
    println(data)
    val df: DataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    println("df: ", df.show())
    val pca: PCAModel = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(3).fit(df)
    val result: DataFrame = pca.transform(df)
    result.select("pcaFeatures").show(false)
  }

}

class FeatureExtractors(spark : SparkSession){

  def getHashFeature(): Unit ={

    val sentenceData: DataFrame = spark.createDataFrame(Seq(
      (0.0, "I go OK OK"),
      (0.0, "you are spark"),
      (1.0, "First you go go start")
    )).toDF("label", "sentence")

    val tokenizer: Tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData: DataFrame = tokenizer.transform(sentenceData)

    val hashingTF: HashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(Math.pow(2, 20).toInt)
    val featurizedData: DataFrame = hashingTF.transform(wordsData)

    val idf: IDF = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel: IDFModel = idf.fit(featurizedData)

    val rescaledData: DataFrame = idfModel.transform(featurizedData)
    rescaledData.select("features").show()
    rescaledData.rdd.foreach(println)
  }

  def getCountVector(): Unit = {

    val df: DataFrame = spark.createDataFrame(Seq(
      (0, Array("a", "b", "c")),
      (1, Array("a", "b", "b", "c", "a"))
    )).toDF("id", "words")

    // 从语料中拟合模型
    val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("words").setOutputCol("features").setVocabSize(3).setMinDF(2).fit(df)
    // 另一种方法，使用字典定义模型
    val cvm: CountVectorizerModel = new CountVectorizerModel(Array("a", "b", "c")).setInputCol("words").setOutputCol("features")

    cvModel.transform(df).show()
    cvm.transform(df).show()
  }

  def getWord2Vec(): Unit ={

    val documentDF: DataFrame = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")
    documentDF.foreach(r => println(r))

    //单词->向量
    val word2vec: Word2Vec = new Word2Vec().setInputCol("text").setOutputCol("result").setVectorSize(10).setMinCount(0)
    val model: Word2VecModel = word2vec.fit(documentDF)

    val result: DataFrame = model.transform(documentDF)
    result.collect().foreach{ case Row(text : Seq[_], features : Vector) =>
      println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n")
    }
  }
}

