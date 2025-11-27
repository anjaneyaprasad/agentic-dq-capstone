package com.anjaneya.dq

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StringType, DoubleType}
import org.slf4j.LoggerFactory

import com.amazon.deequ.checks._
import com.amazon.deequ.VerificationSuite
import com.amazon.deequ.VerificationResult

import com.anjaneya.dq.DqCommon._
import com.anjaneya.dq.DqCommon.DqRuleRow

object DqValidationJob {

  private val logger = LoggerFactory.getLogger(getClass.getName)

  def buildConstraintResultsDf(
      spark: SparkSession,
      checkDf: DataFrame,
      rulesDf: DataFrame,
      runId: String
  ): DataFrame = {
    import spark.implicits._

    val rulesMetaDf = rulesDf.select("RULE_ID", "RULE_TYPE", "COLUMN_NAME")

    val baseResultsDf =
      checkDf
        .withColumn("RUN_ID", lit(runId))
        .withColumn(
          "RULE_ID_EXTRACTED",
          regexp_extract(col("constraint"), "rule_([^\\)]+)", 1)
        )
        .withColumn(
          "RULE_ID",
          regexp_replace(col("RULE_ID_EXTRACTED"), "_(min|max)$", "")
        )
        .withColumn("STATUS", col("constraint_status"))
        .withColumn("MESSAGE", col("constraint_message"))
        .withColumn("CREATED_AT", current_timestamp())
        .select(
          "RUN_ID",
          "RULE_ID",
          "STATUS",
          "MESSAGE",
          "CREATED_AT"
        )

    baseResultsDf
      .join(rulesMetaDf, Seq("RULE_ID"), "left")
      .withColumn("RULE_TYPE", coalesce(col("RULE_TYPE"), lit("UNKNOWN")))
      .withColumn("COLUMN_NAME", col("COLUMN_NAME"))
      .withColumn("METRIC_NAME", lit(null).cast(StringType))
      .withColumn("METRIC_VALUE", lit(null).cast(DoubleType))
      .select(
        "RUN_ID",
        "RULE_ID",
        "RULE_TYPE",
        "COLUMN_NAME",
        "STATUS",
        "MESSAGE",
        "METRIC_NAME",
        "METRIC_VALUE",
        "CREATED_AT"
      )
  }

  def buildBadRowsDf(
      dataDf: DataFrame,
      datasetName: String,
      rule: DqRuleRow,
      runId: String,
      primaryKeyColumn: String
  ): DataFrame = {

    val colName = rule.columnName.get

    val minV = rule.minValue.getOrElse(Double.MinValue)
    val maxV = rule.maxValue.getOrElse(Double.MaxValue)

    dataDf
      .filter(col(colName) < lit(minV) || col(colName) > lit(maxV))
      .limit(100)
      .withColumn("RUN_ID", lit(runId))
      .withColumn("DATASET_NAME", lit(datasetName))
      .withColumn("RULE_ID", lit(rule.ruleId))
      .withColumn("PRIMARY_KEY", col(primaryKeyColumn).cast("string"))
      .withColumn("ROW_JSON", to_json(struct(dataDf.columns.map(col): _*)))
      .withColumn("VIOLATION_MSG", lit(s"$colName out of range [$minV, $maxV]"))
      .withColumn("CREATED_AT", current_timestamp())
      .select(
        "RUN_ID",
        "DATASET_NAME",
        "RULE_ID",
        "PRIMARY_KEY",
        "ROW_JSON",
        "VIOLATION_MSG",
        "CREATED_AT"
      )
  }

  def buildBadRowsForCompleteness(
      dataDf: DataFrame,
      datasetName: String,
      rule: DqRuleRow,
      runId: String,
      primaryKeyColumn: String
  ): DataFrame = {
    val colName = rule.columnName.get

    dataDf
      .filter(col(colName).isNull)
      .limit(100)
      .withColumn("RUN_ID", lit(runId))
      .withColumn("DATASET_NAME", lit(datasetName))
      .withColumn("RULE_ID", lit(rule.ruleId))
      .withColumn("PRIMARY_KEY", col(primaryKeyColumn).cast(StringType))
      .withColumn("ROW_JSON", to_json(struct(dataDf.columns.map(col): _*)))
      .withColumn(
        "VIOLATION_MSG",
        lit(s"$colName is NULL (COMPLETENESS breach)")
      )
      .withColumn("CREATED_AT", current_timestamp())
      .select(
        "RUN_ID",
        "DATASET_NAME",
        "RULE_ID",
        "PRIMARY_KEY",
        "ROW_JSON",
        "VIOLATION_MSG",
        "CREATED_AT"
      )
  }

  def buildBadRowsForUniqueness(
      dataDf: DataFrame,
      datasetName: String,
      rule: DqRuleRow,
      runId: String,
      primaryKeyColumn: String
  ): DataFrame = {
    val colName = rule.columnName.get

    val dupKeysDf = dataDf
      .groupBy(col(colName))
      .count()
      .filter(col("count") > 1)
      .select(col(colName).as("dup_key"))

    dataDf
      .join(dupKeysDf, dataDf(colName) === dupKeysDf("dup_key"), "inner")
      .limit(100)
      .withColumn("RUN_ID", lit(runId))
      .withColumn("DATASET_NAME", lit(datasetName))
      .withColumn("RULE_ID", lit(rule.ruleId))
      .withColumn("PRIMARY_KEY", col(primaryKeyColumn).cast(StringType))
      .withColumn("ROW_JSON", to_json(struct(dataDf.columns.map(col): _*)))
      .withColumn("VIOLATION_MSG", lit(s"$colName violates UNIQUENESS"))
      .withColumn("CREATED_AT", current_timestamp())
      .select(
        "RUN_ID",
        "DATASET_NAME",
        "RULE_ID",
        "PRIMARY_KEY",
        "ROW_JSON",
        "VIOLATION_MSG",
        "CREATED_AT"
      )
  }

  def buildBadRowsForRange(
      dataDf: DataFrame,
      datasetName: String,
      rule: DqRuleRow,
      runId: String,
      primaryKeyColumn: String
  ): DataFrame = {
    val colName = rule.columnName.get
    val minV = rule.minValue.getOrElse(Double.MinValue)
    val maxV = rule.maxValue.getOrElse(Double.MaxValue)

    dataDf
      .filter(col(colName) < lit(minV) || col(colName) > lit(maxV))
      .limit(100)
      .withColumn("RUN_ID", lit(runId))
      .withColumn("DATASET_NAME", lit(datasetName))
      .withColumn("RULE_ID", lit(rule.ruleId))
      .withColumn("PRIMARY_KEY", col(primaryKeyColumn).cast(StringType))
      .withColumn("ROW_JSON", to_json(struct(dataDf.columns.map(col): _*)))
      .withColumn("VIOLATION_MSG", lit(s"$colName out of range [$minV, $maxV]"))
      .withColumn("CREATED_AT", current_timestamp())
      .select(
        "RUN_ID",
        "DATASET_NAME",
        "RULE_ID",
        "PRIMARY_KEY",
        "ROW_JSON",
        "VIOLATION_MSG",
        "CREATED_AT"
      )
  }

  def runValidationForDataset(
      spark: SparkSession,
      datasetName: String,
      batchDate: String
  ): Unit = {

    import spark.implicits._

    logger.info(
      s"[VALIDATION][$datasetName] Starting validation for batchDate=$batchDate"
    )

    // 1) Dataset config
    val dsDf = readSfTable(spark, "DQ_DB.DQ_SCHEMA.DQ_DATASETS")
      .filter(col("DATASET_NAME") === lit(datasetName))

    val dsRowOpt = dsDf.collect().headOption
    if (dsRowOpt.isEmpty) {
      logger.error(
        s"[VALIDATION][$datasetName] Dataset not found in DQ_DATASETS, skipping."
      )
      return
    }
    val dsRow = dsRowOpt.get

    val primaryKeyColumn = dsRow.getAs[String]("PRIMARY_KEY_COLUMN")

    if (primaryKeyColumn == null || primaryKeyColumn.isEmpty) {
      logger.error(
        s"[VALIDATION][$datasetName] No PRIMARY_KEY_COLUMN configured in DQ_DATASETS — cannot continue."
      )
      return
    }

    val objectName = dsRow.getAs[String]("OBJECT_NAME")

    logger.info(s"[VALIDATION][$datasetName] OBJECT_NAME=$objectName")

    // 2) Active ruleset
    val rsDf = readSfTable(spark, "DQ_DB.DQ_SCHEMA.DQ_RULESETS")
      .filter(
        col("DATASET_NAME") === lit(datasetName) && col("IS_ACTIVE") === lit(
          true
        )
      )
      .orderBy(col("VERSION").desc)

    val rsRowOpt = rsDf.collect().headOption
    if (rsRowOpt.isEmpty) {
      logger.warn(
        s"[VALIDATION][$datasetName] No active ruleset found, skipping validation."
      )
      return
    }
    val rsRow = rsRowOpt.get
    val rulesetId = rsRow.getAs[String]("RULESET_ID")

    logger.info(s"[VALIDATION][$datasetName] Using RULESET_ID=$rulesetId")

    // 3) Rules
    val rulesDf = readSfTable(spark, "DQ_DB.DQ_SCHEMA.DQ_RULES")
      .filter(
        col("DATASET_NAME") === lit(datasetName) &&
          col("RULESET_ID") === lit(rulesetId)
      )

    val rules = rulesDf
      .collect()
      .map { r =>
        DqRuleRow(
          ruleId = r.getAs[String]("RULE_ID"),
          ruleType = r.getAs[String]("RULE_TYPE"),
          columnName = Option(r.getAs[String]("COLUMN_NAME")),
          level = Option(r.getAs[String]("LEVEL")),
          threshold =
            Option(r.getAs[java.lang.Double]("THRESHOLD")).map(_.toDouble),
          minValue =
            Option(r.getAs[java.lang.Double]("MIN_VALUE")).map(_.toDouble),
          maxValue =
            Option(r.getAs[java.lang.Double]("MAX_VALUE")).map(_.toDouble),
          allowedValuesJson = Option(r.getAs[String]("ALLOWED_VALUES")),
          paramsJson = Option(r.getAs[String]("PARAMS_JSON"))
        )
      }
      .toSeq

    if (rules.isEmpty) {
      logger.warn(
        s"[VALIDATION][$datasetName] No rules in ruleset=$rulesetId, skipping."
      )
      return
    }

    // 4) Load data
    var dataDf = readSfTable(spark, objectName)

    if (datasetName == "FACT_SALES") {
      dataDf = dataDf.filter(to_date(col("SALE_TS")) === lit(batchDate))
      logger.info(
        s"[VALIDATION][$datasetName] Filtering FACT_SALES on SALE_TS date=$batchDate"
      )
    }

    val totalRows = dataDf.count()
    logger.info(s"[VALIDATION][$datasetName] Loaded rows=$totalRows")

    val message = s"""{"batchDate":"$batchDate"}"""

    if (totalRows == 0) {
      logger.warn(
        s"[VALIDATION][$datasetName] No rows for batchDate=$batchDate, marking run as SKIPPED"
      )

      val runId = java.util.UUID.randomUUID().toString
      val nowTs = java.sql.Timestamp.from(java.time.Instant.now())
      // val now   = java.time.Instant.now().toString

      val runsDf = Seq(
        (
          runId,
          datasetName,
          rulesetId,
          "VALIDATION",
          nowTs,
          nowTs,
          "SKIPPED",
          0L,
          0L,
          message
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

      writeSfTable(runsDf, "DQ_DB.DQ_SCHEMA.DQ_RUNS")
      return
    }

    val runId = java.util.UUID.randomUUID().toString
    // val runStart = java.time.Instant.now().toString
    val runStartTs = java.sql.Timestamp.from(java.time.Instant.now())

    // 5) Insert initial RUN row (RUNNING)
    val initRunDf = Seq(
      (
        runId,
        datasetName,
        rulesetId,
        "VALIDATION",
        runStartTs,
        null.asInstanceOf[java.sql.Timestamp], // FINISHED_AT
        "RUNNING",
        totalRows,
        null.asInstanceOf[java.lang.Long],
        message
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

    writeSfTable(initRunDf, "DQ_DB.DQ_SCHEMA.DQ_RUNS")

    // 6) Run Deequ verification
    val check = buildCheckForDataset(datasetName, rules)

    val verificationResult: VerificationResult = VerificationSuite()
      .onData(dataDf)
      .addCheck(check)
      .run()

    val checkDf =
      VerificationResult.checkResultsAsDataFrame(spark, verificationResult)

    val rulesMetaDf = rulesDf.select("RULE_ID", "RULE_TYPE", "COLUMN_NAME")

    // Raw Deequ check results
    val baseResultsDf =
      checkDf
        .withColumn("RUN_ID", lit(runId))
        .withColumn(
          "RULE_ID",
          regexp_extract(col("constraint"), "rule_(.+)", 1)
        )
        .withColumn("STATUS", col("constraint_status"))
        .withColumn("MESSAGE", col("constraint_message"))
        .withColumn("CREATED_AT", current_timestamp())
        .select(
          "RUN_ID",
          "RULE_ID",
          "STATUS",
          "MESSAGE",
          "CREATED_AT"
        )

    val constraintResultsDf =
      buildConstraintResultsDf(spark, checkDf, rulesDf, runId)

    writeSfTable(constraintResultsDf, "DQ_DB.DQ_SCHEMA.DQ_CONSTRAINT_RESULTS")

    val failedRules =
      constraintResultsDf.filter(col("STATUS") === "Failure").count()
    logger.info(s"[VALIDATION][$datasetName] Failed rules count=$failedRules")

    // 7) Example bad rows capture: RANGE rules
    val rangeRules = rules.filter(r =>
      r.ruleType.equalsIgnoreCase("RANGE") && r.columnName.isDefined
    )

    rangeRules.foreach { rule =>
      val badRowsDf =
        buildBadRowsForRange(dataDf, datasetName, rule, runId, primaryKeyColumn)
      if (!badRowsDf.rdd.isEmpty()) {
        writeSfTable(badRowsDf, "DQ_DB.DQ_SCHEMA.DQ_BAD_ROWS")
      }
    }

    // 7a) Bad rows for COMPLETENESS rules (null values)
    val completenessRules =
      rules.filter(r =>
        r.ruleType.equalsIgnoreCase("COMPLETENESS") && r.columnName.isDefined
      )

    completenessRules.foreach { rule =>
      val badRowsDf = buildBadRowsForCompleteness(
        dataDf,
        datasetName,
        rule,
        runId,
        primaryKeyColumn
      )
      if (!badRowsDf.rdd.isEmpty()) {
        writeSfTable(badRowsDf, "DQ_DB.DQ_SCHEMA.DQ_BAD_ROWS")
      }
    }

// 7b) Bad rows for UNIQUENESS rules (duplicate values)
    val uniquenessRules =
      rules.filter(r =>
        r.ruleType.equalsIgnoreCase("UNIQUENESS") && r.columnName.isDefined
      )

    uniquenessRules.foreach { rule =>
      val badRowsDf = buildBadRowsForUniqueness(
        dataDf,
        datasetName,
        rule,
        runId,
        primaryKeyColumn
      )
      if (!badRowsDf.rdd.isEmpty()) {
        writeSfTable(badRowsDf, "DQ_DB.DQ_SCHEMA.DQ_BAD_ROWS")
      }
    }

    // 8) Final RUN row (we append final state – in real life you’d MERGE)
    val status =
      if (failedRules > 0) "COMPLETED_WITH_ERRORS" else "SUCCESS"

    // val runEnd = java.time.Instant.now().toString
    val runEndTs = java.sql.Timestamp.from(java.time.Instant.now())

    val finalRunDf = Seq(
      (
        runId,
        datasetName,
        rulesetId,
        "VALIDATION",
        runStartTs,
        runEndTs,
        status,
        totalRows,
        failedRules,
        message
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

    writeSfTable(finalRunDf, "DQ_DB.DQ_SCHEMA.DQ_RUNS")

    logger.info(
      s"[VALIDATION][$datasetName] Completed. status=$status, RUN_ID=$runId  for batchDate=$batchDate"
    )
  }

  def main(args: Array[String]): Unit = {
    val master = sys.env.getOrElse("SPARK_MASTER", "local[*]")

    val spark = SparkSession
      .builder()
      .appName("DQ Job - VALIDATION")
      .master(master)
      .config("spark.sql.shuffle.partitions", "8")
      .getOrCreate()

    spark.conf.set("spark.sql.codegen.wholeStage", "false")

    val allDatasets =
      Seq("DIM_CUSTOMERS", "DIM_PRODUCTS", "DIM_STORES", "FACT_SALES")

    // Arg handling:
    // 0 args  → all datasets, batchDate = today
    // 1 arg   → dataset = arg(0), batchDate = today
    // 2+ args → dataset = arg(0), batchDate = arg(1)
    val (datasets: Seq[String], batchDate: String) =
      if (args.length >= 2) {
        (Seq(args(0).toUpperCase), args(1))
      } else if (args.length == 1) {
        (Seq(args(0).toUpperCase), java.time.LocalDate.now().toString)
      } else {
        (allDatasets, java.time.LocalDate.now().toString)
      }

    logger.info(
      s"[VALIDATION] Starting validation for datasets=${datasets.mkString(",")} batchDate=$batchDate"
    )

    datasets.foreach { ds =>
      try {
        runValidationForDataset(spark, ds, batchDate)
      } catch {
        case e: Throwable =>
          logger.error(s"[VALIDATION][$ds] Failed for batchDate=$batchDate", e)
      }
    }

    logger.info("[VALIDATION] All datasets processed, stopping SparkSession")
    spark.stop()
  }
}
