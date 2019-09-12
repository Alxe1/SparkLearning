import scala.collection.mutable

object Test {
  def main(args: Array[String]): Unit = {

//    val lists = List(1,2,3)
//    val list2: List[Int] = lists :+ 4
//    println(list2)
//
//    val list3 = 2::List(3, 7)::lists
//    println(list3)
//
//    println(List(3, 7) ::: lists)
//
//    val map: Map[String, Int] = Map("a" -> 1, "b" -> 4)
//    println(map.get("a").getOrElse(0))

    this.test()

  }

  def test(): Unit ={
    val map = new mutable.HashMap[String, Int]
    map += (("a", 1))
    map += (("b", 2))
    println(map.getOrElse("a", 0))

  }
}
