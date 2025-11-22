package com.anjaneya.dq.jobs

import com.anjaneya.dq.util.SparkSessionBuilder
import org.apache.spark.sql.functions._

object GenerateSampleSalesData {

  def main(args: Array[String]): Unit = {
    val spark = SparkSessionBuilder.build("GenerateSampleSalesData")
    import spark.implicits._

    try {
      val df = Seq(
        ("O1", "C1", 100.5, "INR", "2025-11-18"),
        ("O2", "C2", 200.0, "USD", "2025-11-18"),
        ("O3", "C3", -10.0, "EUR", "2025-11-19") // one bad row for DQ failure
      ).toDF("order_id", "customer_id", "net_amount", "currency", "sale_date")

      df.show(false)

      // This path must match your datasets.yaml table_or_path
      df.write.mode("overwrite").parquet("data/sales_fact_daily")

      println("Sample data written to data/sales_fact_daily")

    } finally {
      spark.stop()
    }
  }
}