package com.anjaneya.dq

import org.apache.spark.sql.SparkSession
import org.scalatest.Suite

trait SparkTestSession { self: Suite =>

  // Just exposes the shared SparkSession
  protected lazy val spark: SparkSession = TestSpark.spark
}
