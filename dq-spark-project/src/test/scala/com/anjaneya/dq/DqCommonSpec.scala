package com.anjaneya.dq

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import com.amazon.deequ.VerificationSuite
import com.amazon.deequ.VerificationResult
import com.amazon.deequ.checks.CheckStatus

class DqCommonSpec extends AnyFunSuite with SparkTestSession {

  import spark.implicits._
  import DqCommon._

  test(
    "buildCheckForDataset should create rules that detect completeness and range violations"
  ) {

    val df = Seq(
      ("1", "Alice", 100.0),
      ("2", null.asInstanceOf[String], 50.0),
      ("3", "Charlie", 999999.0) // outlier
    ).toDF("ID", "NAME", "AMOUNT")

    val rules = Seq(
      DqRuleRow(
        ruleId = "R1",
        ruleType = "COMPLETENESS",
        columnName = Some("NAME"),
        level = Some("ERROR"),
        threshold = Some(0.9),
        minValue = None,
        maxValue = None,
        allowedValuesJson = None,
        paramsJson = None
      ),
      DqRuleRow(
        ruleId = "R2",
        ruleType = "RANGE",
        columnName = Some("AMOUNT"),
        level = Some("ERROR"),
        threshold = None,
        minValue = Some(0.0),
        maxValue = Some(1000.0),
        allowedValuesJson = None,
        paramsJson = None
      )
    )

    val check = buildCheckForDataset("TEST_DATASET", rules)

    val result: VerificationResult = VerificationSuite()
      .onData(df)
      .addCheck(check)
      .run()

    // 1) Overall check should *not* be Success
    import com.amazon.deequ.checks.CheckStatus
    assert(result.status != CheckStatus.Success)

    // 2) Look at constraint-level results
    val resultDf = VerificationResult.checkResultsAsDataFrame(spark, result)

    // Just for sanity while debugging:
    // resultDf.show(truncate = false)

    // We don't rely on rule ids anymore, just on Deequ's own descriptions
    val rows = resultDf
      .select("constraint", "constraint_status")
      .as[(String, String)]
      .collect()
      .toSeq

    // at least one completeness-related failure
    val completenessFailed =
      rows.exists { case (c, st) =>
        c.toLowerCase.contains("completeness") && st == "Failure"
      }

    // at least one range-related failure (min or max)
    val rangeFailed =
      rows.exists { case (c, st) =>
        (c.toLowerCase.contains("minimum") || c.toLowerCase.contains(
          "maximum"
        )) &&
        st == "Failure"
      }

    assert(completenessFailed, "Expected a completeness failure on NAME")
    assert(rangeFailed, "Expected a range failure on AMOUNT")
  }
}
