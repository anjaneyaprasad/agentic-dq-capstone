package com.anjaneya.dq.util

import org.apache.spark.sql.SparkSession

object SparkSessionBuilder {

  def build(appName: String): SparkSession = {
    SparkSession.builder()
      .appName(appName)
      .master(sys.props.getOrElse("spark.master", "local[*]"))
      .getOrCreate()
  }
}