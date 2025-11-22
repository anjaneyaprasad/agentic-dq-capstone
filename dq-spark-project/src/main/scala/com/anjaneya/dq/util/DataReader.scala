package com.anjaneya.dq.util

import com.anjaneya.dq.config.DatasetConfig
import org.apache.spark.sql.{DataFrame, SparkSession}

object DataReader {

  def readDataset(spark: SparkSession, ds: DatasetConfig): DataFrame = {
    ds.source.toLowerCase match {
      case "parquet" | "s3-parquet" =>
        spark.read.parquet(ds.table_or_path)

      case "csv" =>
        spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .csv(ds.table_or_path)

      // placeholder to extend it for "snowflake", "delta" etc.

      case other =>
        throw new IllegalArgumentException(s"Unsupported source: $other")
    }
  }
}