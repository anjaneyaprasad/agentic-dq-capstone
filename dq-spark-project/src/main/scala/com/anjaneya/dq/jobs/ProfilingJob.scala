package com.anjaneya.dq.jobs

import com.anjaneya.dq.util.{ConfigLoader, DataReader, SparkSessionBuilder}
import com.anjaneya.dq.config.DatasetConfig

import com.amazon.deequ.analyzers._
import com.amazon.deequ.analyzers.runners.{AnalysisRunner, AnalyzerContext}
import com.amazon.deequ.metrics.Metric

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object ProfilingJob {

  // Usage: ProfilingJob <datasetName> 
  // This job reads the dataset config from datasets.yaml, loads the data, 
  // runs profiling analysis, and writes metrics to output.
  // The output path is hardcoded to "output/dq_metrics_all" for simplicity.
  // TODO: make it configurable.

  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      println("Usage: ProfilingJob <datasetName>")
      System.exit(1)
    }

    val datasetName = args(0)
    val spark = SparkSessionBuilder.build(s"ProfilingJob-$datasetName")
    import spark.implicits._

    try {
      val cfg = ConfigLoader.loadDatasetsConfig()
      val ds = ConfigLoader.getDatasetConfig(datasetName, cfg)

      val df = DataReader.readDataset(spark, ds)

      val result: AnalyzerContext = runAnalysis(df, ds)

      val metricsRawDf = AnalyzerContext.successMetricsAsDataFrame(spark, result)

      val metricsDf = metricsRawDf
      .withColumn("dataset_name", lit(ds.name))
      .withColumn("run_ts", current_timestamp())
      .select(
        col("dataset_name"),
        col("run_ts"),
        col("entity"),
        col("name"),
        col("instance"),
        col("value")
        )
        
      metricsDf.show(false)

      metricsDf
      .write
      .mode("append")
      .parquet("output/dq_metrics_all")
      // TODO: make output path configurable or may be write to a database
      // Then for reporting, we can read from that database or output location, bypassing HTML report generation.

    } catch {
      case ex: Exception =>
        println(s"Error in ProfilingJob for dataset $datasetName: ${ex.getMessage}")
        ex.printStackTrace()
        
    } finally {
      spark.stop()
    }
  }

  private def runAnalysis(df: DataFrame, ds: DatasetConfig): AnalyzerContext = {

    val baseAnalyzers: Seq[Analyzer[_, Metric[_]]] =
      Seq(Size())

    val criticalAnalyzers: Seq[Analyzer[_, Metric[_]]] =
      ds.critical_columns.flatMap { colName =>
        Seq[Analyzer[_, Metric[_]]](
          Completeness(colName),
          ApproxCountDistinct(colName)   // ApproxDistinctCount
        )
      }

    val extraAnalyzers: Seq[Analyzer[_, Metric[_]]] =
      Seq(
        Mean("net_amount"),
        StandardDeviation("net_amount"), // StdDev
        Minimum("net_amount"),           // Min
        Maximum("net_amount")            // Max
      )

    AnalysisRunner
      .onData(df)
      .addAnalyzers(baseAnalyzers ++ criticalAnalyzers ++ extraAnalyzers)
      .run()
  }
}
