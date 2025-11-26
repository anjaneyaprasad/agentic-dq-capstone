package com.anjaneya.dq

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.slf4j.LoggerFactory

import com.amazon.deequ.checks.{Check, CheckLevel}
import com.amazon.deequ.metrics.Metric
import com.amazon.deequ.analyzers.Analyzer

import scala.util.Try
import scala.util.matching.Regex

object DqCommon {

  private val logger = LoggerFactory.getLogger(getClass.getName)

  // ---------------- Snowflake helpers ----------------

  def sfOptions: Map[String, String] = {
    val opts = Map(
      "sfURL" -> sys.env("SNOWFLAKE_URL"),
      "sfUser" -> sys.env("SNOWFLAKE_USER"),
      "sfPassword" -> sys.env("SNOWFLAKE_PASSWORD"),
      "sfWarehouse" -> sys.env("SNOWFLAKE_WAREHOUSE"),
      "sfDatabase" -> "DQ_DB",
      "sfSchema" -> "DQ_SCHEMA"
    )
    logger.info(
      s"[DqCommon] Using Snowflake options for database=DQ_DB schema=DQ_SCHEMA"
    )
    opts
  }

  def readSfTable(spark: SparkSession, dbtable: String): DataFrame = {
    logger.debug(s"[DqCommon] Reading from Snowflake table: $dbtable")
    spark.read
      .format("snowflake")
      .options(sfOptions)
      .option("dbtable", dbtable)
      .load()
  }

  def writeSfTable(
      df: DataFrame,
      dbtable: String,
      mode: String = "append"
  ): Unit = {
    val cnt = Try(df.count()).getOrElse(-1L)
    logger.debug(
      s"[DqCommon] Writing to Snowflake table: $dbtable, mode=$mode, rows=$cnt"
    )
    df.write
      .format("snowflake")
      .options(sfOptions)
      .option("dbtable", dbtable)
      .mode(mode)
      .save()
  }

  // ---------------- Rule model ----------------

  case class DqRuleRow(
      ruleId: String,
      ruleType: String,
      columnName: Option[String],
      level: Option[String],
      threshold: Option[Double],
      minValue: Option[Double],
      maxValue: Option[Double],
      allowedValuesJson: Option[String],
      paramsJson: Option[String]
  )

  // ---------------- Deequ helpers ----------------

  private def addCompleteness(
      check: Check,
      column: String,
      threshold: Double,
      constraintName: String
  ): Check = {
    logger.debug(
      s"[DqCommon] COMPLETENESS rule=$constraintName col=$column thr=$threshold"
    )
    check.hasCompleteness(
      column,
      _ >= threshold,
      hint = Some(constraintName)
    )
  }

  private def addUniqueness(
      check: Check,
      column: String,
      constraintName: String
  ): Check = {
    logger.debug(s"[DqCommon] UNIQUENESS rule=$constraintName col=$column")
    check.isUnique(
      column,
      hint = Some(constraintName)
    )
  }

  private def addRange(
      check: Check,
      column: String,
      min: Option[Double],
      max: Option[Double],
      baseName: String
  ): Check = {
    var c = check

    min.foreach { m =>
      logger.debug(
        s"[DqCommon] RANGE-MIN rule=${baseName}_min col=$column min=$m"
      )
      c = c.hasMin(
        column,
        _ >= m,
        hint = Some(s"${baseName}_min")
      )
    }

    max.foreach { M =>
      logger.debug(
        s"[DqCommon] RANGE-MAX rule=${baseName}_max col=$column max=$M"
      )
      c = c.hasMax(
        column,
        _ <= M,
        hint = Some(s"${baseName}_max")
      )
    }

    c
  }

  private def addPattern(
      check: Check,
      column: String,
      pattern: String,
      constraintName: String
  ): Check = {
    val regex: Regex = pattern.r
    logger.debug(
      s"[DqCommon] PATTERN rule=$constraintName col=$column pattern=$pattern"
    )
    // Deequ 2.x: hasPattern(col, regex, assertion, name, hint)
    check.hasPattern(
      column,
      regex,
      assertion = Check.IsOne,
      name = Some(constraintName),
      hint = None
    )
  }

  /** dispatcher: turns one DQ_RULE row into Deequ constraints */
  private def addRuleFromMeta(
      check: Check,
      rule: DqRuleRow
  ): Check = {
    val colOpt = rule.columnName
    val constraintName = s"rule_${rule.ruleId}"

    rule.ruleType.toUpperCase match {
      case "COMPLETENESS" =>
        val col = colOpt.getOrElse(
          throw new IllegalArgumentException(
            s"COMPLETENESS rule requires COLUMN_NAME, ruleId=${rule.ruleId}"
          )
        )
        val th = rule.threshold.getOrElse(1.0)
        addCompleteness(check, col, th, constraintName)

      case "UNIQUENESS" | "IS_UNIQUE" =>
        val col = colOpt.getOrElse(
          throw new IllegalArgumentException(
            s"UNIQUENESS rule requires COLUMN_NAME, ruleId=${rule.ruleId}"
          )
        )
        addUniqueness(check, col, constraintName)

      case "RANGE" =>
        val col = colOpt.getOrElse(
          throw new IllegalArgumentException(
            s"RANGE rule requires COLUMN_NAME, ruleId=${rule.ruleId}"
          )
        )
        addRange(check, col, rule.minValue, rule.maxValue, constraintName)

      case "PATTERN" | "REGEX" =>
        val col = colOpt.getOrElse(
          throw new IllegalArgumentException(
            s"PATTERN rule requires COLUMN_NAME, ruleId=${rule.ruleId}"
          )
        )
        val pattern: String =
          rule.paramsJson
            .flatMap { js =>
              // naive parse of {"pattern":"^ABC.*"} – fine for your own generated JSON
              Try {
                val idx = js.indexOf("pattern")
                if (idx >= 0) {
                  val after = js.substring(idx)
                  val firstQuote = after.indexOf('"', after.indexOf(':'))
                  val secondQuote = after.indexOf('"', firstQuote + 1)
                  after.substring(firstQuote + 1, secondQuote)
                } else ".*"
              }.toOption
            }
            .getOrElse(".*")
        addPattern(check, col, pattern, constraintName)

      case other =>
        logger.warn(
          s"[DqCommon] Unsupported RULE_TYPE=$other for ruleId=${rule.ruleId}, skipping."
        )
        check
    }
  }

  /** Public API used by DqValidationJob */
  def buildCheckForDataset(
      datasetName: String,
      rules: Seq[DqRuleRow]
  ): Check = {
    logger.info(
      s"[DqCommon] Building Check for dataset=$datasetName, rules=${rules.size}"
    )
    rules.foldLeft(Check(CheckLevel.Error, s"DQ-$datasetName")) { (chk, r) =>
      addRuleFromMeta(chk, r)
    }
  }
}
