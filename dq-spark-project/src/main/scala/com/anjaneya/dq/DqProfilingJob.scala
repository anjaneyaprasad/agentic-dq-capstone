package com.anjaneya.dq

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.slf4j.LoggerFactory

import com.amazon.deequ.analyzers._
import com.amazon.deequ.analyzers.runners.{AnalysisRunner, AnalyzerContext}
import com.amazon.deequ.metrics.Metric

import com.anjaneya.dq.DqCommon._

object DqProfilingJob {

  private val logger = LoggerFactory.getLogger(getClass.getName)

  // ------------ Helper 1: analyzers builder (used by tests + runProfilingForDataset) ------------

  /** Build profiling analyzers for a dataframe, based on its schema. */
  def buildProfilingAnalyzers(df: DataFrame): Seq[Analyzer[_, Metric[_]]] = {
    val allCols = df.columns.toSeq
    val schemaByCol = df.schema.map(f => f.name -> f.dataType).toMap

    val numericCols: Seq[String] = allCols.filter { c =>
      schemaByCol.get(c).exists {
        case _: ByteType    => true
        case _: ShortType   => true
        case _: IntegerType => true
        case _: LongType    => true
        case _: FloatType   => true
        case _: DoubleType  => true
        case _: DecimalType => true
        case _              => false
      }
    }

    // 1) Basic dataset-level metric
    val basic: Seq[Analyzer[_, Metric[_]]] =
      Seq(Size(): Analyzer[_, Metric[_]])

    // 2) Column-level analyzers for ALL columns
    val allColumnAnalyzers: Seq[Analyzer[_, Metric[_]]] =
      allCols.flatMap { c =>
        Seq[Analyzer[_, Metric[_]]](
          Completeness(c),
          ApproxCountDistinct(c),
          com.amazon.deequ.analyzers
            .DataType(c) // explicitly Deequ's DataType analyzer
        )
      }

    // 3) Numeric-only analyzers
    val numericAnalyzers: Seq[Analyzer[_, Metric[_]]] =
      numericCols.flatMap { c =>
        Seq[Analyzer[_, Metric[_]]](
          Minimum(c),
          Maximum(c),
          Mean(c)
        )
      }

    basic ++ allColumnAnalyzers ++ numericAnalyzers
  }

  // ------------ Helper 2: datasetsFromMetaDf (used by tests, optional for main) ------------

  // Pure-ish helper to derive active dataset names from DQ_DATASETS-like DF.

  def datasetsFromMetaDf(metaDf: DataFrame): Seq[String] = {
    import metaDf.sparkSession.implicits._

    metaDf
      .filter(col("IS_ACTIVE") === lit(true))
      .select("DATASET_NAME")
      .distinct()
      .as[String]
      .collect()
      .sorted
  }

  // ------------ Main profiling logic for a single dataset ------------

  def runProfilingForDataset(
      spark: SparkSession,
      datasetName: String
  ): Unit = {
    import spark.implicits._

    logger.info(s"[PROFILING][$datasetName] Starting profiling")

    // 1) Find dataset metadata (Snowflake table name)
    val dsDf = readSfTable(spark, "DQ_DB.DQ_SCHEMA.DQ_DATASETS")
      .filter(col("DATASET_NAME") === lit(datasetName))

    val dsRowOpt = dsDf.collect().headOption
    if (dsRowOpt.isEmpty) {
      logger.warn(
        s"[PROFILING][$datasetName] Not found in DQ_DATASETS, skipping."
      )
      return
    }

    val dsRow = dsRowOpt.get
    val objectName = dsRow.getAs[String]("OBJECT_NAME")

    logger.info(s"[PROFILING][$datasetName] OBJECT_NAME=$objectName")

    // 2) Load data from Snowflake
    val dataDf = readSfTable(spark, objectName)
    val totalRows = dataDf.count()
    logger.info(s"[PROFILING][$datasetName] Loaded rows=$totalRows")

    if (totalRows == 0) {
      logger.warn(s"[PROFILING][$datasetName] No rows, skipping profiling")
      return
    }

    // 3) Build analyzer list using the helper (so tests cover this logic)
    val analyzers: Seq[Analyzer[_, Metric[_]]] = buildProfilingAnalyzers(dataDf)

    logger.info(
      s"[PROFILING][$datasetName] Running ${analyzers.size} analyzers"
    )

    // 4) Run analysis
    val analysisResult: AnalyzerContext = AnalysisRunner
      .onData(dataDf)
      .addAnalyzers(analyzers)
      .run()

    val metricsDf =
      AnalyzerContext.successMetricsAsDataFrame(spark, analysisResult)
    logger.info(
      s"[PROFILING][$datasetName] Deequ metrics rows=${metricsDf.count()}"
    )

    // 5) Map to DQ_PROFILING_METRICS schema
    val profilingDf: DataFrame =
      metricsDf
        .filter(col("entity") === lit("Column"))
        .withColumn("DATASET_NAME", lit(datasetName))
        .withColumn("COLUMN_NAME", col("instance"))
        .withColumn("METRIC_NAME", col("name"))
        .withColumn("METRIC_VALUE", col("value").cast(DoubleType))
        .withColumn("RUN_TS", current_timestamp())
        .select(
          "DATASET_NAME",
          "COLUMN_NAME",
          "METRIC_NAME",
          "METRIC_VALUE",
          "RUN_TS"
        )

    logger.info(
      s"[PROFILING][$datasetName] Writing ${profilingDf.count()} rows to DQ_PROFILING_METRICS"
    )
    profilingDf.printSchema()

    writeSfTable(profilingDf, "DQ_DB.DQ_SCHEMA.DQ_PROFILING_METRICS")

    val runId = java.util.UUID.randomUUID().toString
    val startTs = java.sql.Timestamp.from(java.time.Instant.now())

    val runLogDf = Seq(
      (
        runId,
        datasetName,
        null.asInstanceOf[String], // RULESET_ID
        "PROFILING",
        startTs,
        java.sql.Timestamp.from(java.time.Instant.now()),
        "SUCCESS",
        totalRows,
        null.asInstanceOf[java.lang.Long],
        null.asInstanceOf[String]
      )
    ).toDF(
      "RUN_ID",
      "DATASET_NAME",
      "RULESET_ID",
      "RUN_TYPE",
      "STARTED_AT",
      "FINISHED_AT",
      "STATUS",
      "TOTAL_ROWS",
      "FAILED_ROWS",
      "MESSAGE"
    )

    writeSfTable(runLogDf, "DQ_DB.DQ_SCHEMA.DQ_RUNS")

    logger.info(s"[PROFILING][$datasetName] Completed run successfully")
  }

  // ------------ Main entry point ------------

  def main(args: Array[String]): Unit = {
    val master = sys.env.getOrElse("SPARK_MASTER", "local[*]")

    val spark = SparkSession
      .builder()
      .appName("DQ Job - PROFILING")
      .master(master)
      .config("spark.sql.shuffle.partitions", "8")
      .getOrCreate()

    spark.conf.set("spark.sql.codegen.wholeStage", "false")

    val allDatasets =
      Seq("DIM_CUSTOMERS", "DIM_PRODUCTS", "DIM_STORES", "FACT_SALES")

    // If a dataset is passed as arg(0) → run only that one (from UI)
    //  Else → run all datasets (CLI / batch mode)
    val datasets: Seq[String] =
      if (args.nonEmpty) Seq(args(0).toUpperCase)
      else allDatasets

    datasets.foreach { ds =>
      try {
        runProfilingForDataset(spark, ds)
      } catch {
        case e: Throwable =>
          logger.error(s"[PROFILING][$ds] Failed", e)
      }
    }

    logger.info("[PROFILING] All datasets processed, stopping SparkSession")
    spark.stop()
  }
}
