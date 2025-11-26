package com.anjaneya.dq

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

class DqValidationJobSpec extends AnyFunSuite with SparkTestSession {

  test("Range bad-row filter should catch out-of-range AMOUNT values") {
    import spark.implicits._
    val df = Seq(
      ("1", 10.0),
      ("2", 500.0),
      ("3", -5.0),
      ("4", 10000.0)
    ).toDF("ID", "AMOUNT")

    val minV = 0.0
    val maxV = 1000.0

    val badRowsDf =
      df.filter(col("AMOUNT") < lit(minV) || col("AMOUNT") > lit(maxV))
    val badIds = badRowsDf.select("ID").as[String].collect().toSet

    assert(badIds == Set("3", "4"))
  }
}
