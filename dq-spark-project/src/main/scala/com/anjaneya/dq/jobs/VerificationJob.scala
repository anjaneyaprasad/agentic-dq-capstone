package com.anjaneya.dq.jobs

import com.anjaneya.dq.util.{ConfigLoader, DataReader, SparkSessionBuilder, ConstraintsLoader}
import com.anjaneya.dq.config.{DatasetConstraintsConfig, ConstraintRule}

import com.amazon.deequ.VerificationSuite
import com.amazon.deequ.checks.{Check, CheckLevel, CheckStatus}
import com.amazon.deequ.VerificationResult

import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

object VerificationJob {

  // Usage: VerificationJob <datasetName>
  // This job reads the dataset config from datasets.yaml, loads the data,
  // loads the constraints YAML for that dataset, builds Deequ Checks,
  // runs the verification, and writes results to output.
  // The output path is hardcoded to "output/dq_verification_all" for simplicity.
  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      println("Usage: VerificationJob <datasetName>")
      System.exit(1)
    }

    val datasetName = args(0)
    val spark = SparkSessionBuilder.build(s"VerificationJob-$datasetName")

    import spark.implicits._

    try {
      val cfg = ConfigLoader.loadDatasetsConfig()
      val ds  = ConfigLoader.getDatasetConfig(datasetName, cfg)

      val df = DataReader.readDataset(spark, ds)

      // load YAML constraints for this dataset
      val constraintsCfg = ConstraintsLoader.loadConstraintsFor(ds.name)

      // build Deequ Checks from config
      val checks: Seq[Check] = buildChecksFromConfig(ds.name, constraintsCfg)

      // run all checks
      val verificationRun = checks.foldLeft(VerificationSuite().onData(df)) {
        case (suite, check) => suite.addCheck(check)
      }.run()

      val resultDf = VerificationResult
        .checkResultsAsDataFrame(spark, verificationRun)
        .withColumn("dataset_name", lit(ds.name))
        .withColumn("run_ts", current_timestamp())
        .select(
          col("dataset_name"),
          col("run_ts"),
          col("check"),
          col("check_level"),
          col("constraint"),
          col("constraint_status"),
          col("constraint_message")
        )

      resultDf.show(false)

      resultDf
        .write
        .mode("append")
        .parquet("output/dq_verification_all") // can be made configurable or write to DB

      val hasFailures = verificationRun.checkResults.exists {
        case (_, cr) => cr.status != CheckStatus.Success
      }

      if (hasFailures) System.exit(2)
    } catch {
      case ex: Exception =>
        println(s"Error in VerificationJob for dataset $datasetName: ${ex.getMessage}")
        ex.printStackTrace()

    } finally {
      spark.stop()
    }
  }

  /**
    * Convert DatasetConstraintsConfig into one or more Deequ Checks.
    * We group by level (ERROR/WARN) so each level becomes a separate Check.
    */
  private def buildChecksFromConfig(
    datasetName: String,
    cfg: DatasetConstraintsConfig
  ): Seq[Check] = {

    val byLevel: Map[String, Seq[ConstraintRule]] =
      cfg.constraints.groupBy(_.level.toUpperCase)

    byLevel.toSeq.map { case (levelStr, rules) =>
      val level = levelStr match {
        case "ERROR" => CheckLevel.Error
        case "WARN"  => CheckLevel.Warning
        case other =>
          throw new IllegalArgumentException(s"Unsupported check level: $other")
      }

      val baseName = s"$datasetName-$levelStr-constraints"

      rules.foldLeft(Check(level, baseName)) { (check, rule) =>
        rule.`type`.toLowerCase match {
          // ---------- Completeness ----------
          case "completeness" =>
            check.isComplete(rule.column)

          case "completeness_threshold" =>
            val thr = rule.threshold.getOrElse {
              throw new IllegalArgumentException(
                s"completeness_threshold for ${rule.column} requires 'threshold'"
              )
            }
            check.hasCompleteness(rule.column, _ >= thr)

          // ---------- Non-negative ----------
          case "non_negative" | "nonnegative" =>
            check.isNonNegative(rule.column)

          // ---------- Domain / allowed values ----------
          case "domain" | "inclusion" =>
            if (rule.allowed_values.isEmpty) {
              throw new IllegalArgumentException(
                s"Constraint 'domain' on column ${rule.column} requires 'allowed_values'"
              )
            }
            check.isContainedIn(rule.column, rule.allowed_values.toArray)

          // ---------- Uniqueness ----------
          case "unique" =>
            check.isUnique(rule.column)

          // ---------- Dataset size ----------
          case "size_greater_than" =>
            val thr = rule.threshold.getOrElse {
              throw new IllegalArgumentException(
                s"size_greater_than requires 'threshold'"
              )
            }
            check.hasSize(_ >= thr.toLong)

          // ---------- Min / Max value constraints ----------
          case "min_value" =>
            val minVal = rule.min.getOrElse {
              throw new IllegalArgumentException(
                s"min_value for ${rule.column} requires 'min'"
              )
            }
            check.hasMin(rule.column, _ >= minVal)

          case "max_value" =>
            val maxVal = rule.max.getOrElse {
              throw new IllegalArgumentException(
                s"max_value for ${rule.column} requires 'max'"
              )
            }
            check.hasMax(rule.column, _ <= maxVal)

          // ---------- Unsupported ----------
          case other =>
            throw new IllegalArgumentException(
              s"Unsupported constraint type: $other for column ${rule.column}"
            )
        }
      }
    }
  }
}