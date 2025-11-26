package com.anjaneya.dq

import org.apache.spark.sql.SparkSession

object TestSpark {

  // Single, shared SparkSession for all tests
  lazy val spark: SparkSession =
    SparkSession.builder()
      .appName("dq-test-suite")
      .master("local[*]")
      .config("spark.ui.enabled", "false")
      // add any test-specific configs here if needed
      .getOrCreate()
}