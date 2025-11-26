package com.anjaneya.dq

import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class DqConstraintResultsMappingSpec extends AnyFunSuite with SparkTestSession {

  test(
    "buildConstraintResultsDf should match DQ_CONSTRAINT_RESULTS schema and fill RULE_TYPE"
  ) {

    import spark.implicits._

    // --- fake Deequ checkDf: just the columns we actually use ---
    val checkDf = Seq(
      (
        "Check for DIM_CUSTOMERS",
        "Error",
        "Error",
        "CompletenessConstraint(rule_R1)",
        "Failure",
        "msg1"
      ),
      (
        "Check for DIM_CUSTOMERS",
        "Error",
        "Error",
        "RangeConstraint(rule_R2)",
        "Success",
        "msg2"
      )
    ).toDF(
      "checkDescription",
      "checkLevel",
      "checkStatus",
      "constraint",
      "constraintStatus",
      "constraintMessage"
    )

    // mimic what production code does to Deequ's DF
    val normalizedCheckDf = checkDf.select(
      col("checkDescription").as("check"),
      col("checkLevel").as("check_level"),
      col("checkStatus").as("check_status"),
      col("constraint"),
      col("constraintStatus").as("constraint_status"),
      col("constraintMessage").as("constraint_message")
    )

    // --- fake rulesDf from DQ_RULES ---
    val rulesDf = Seq(
      ("R1", "COMPLETENESS", "CUSTOMER_ID"),
      ("R2", "RANGE", "AMOUNT")
    ).toDF("RULE_ID", "RULE_TYPE", "COLUMN_NAME")

    val runId = "TEST_RUN"

    val resDf: DataFrame =
      DqValidationJob.buildConstraintResultsDf(
        spark,
        normalizedCheckDf,
        rulesDf,
        runId
      )

    // --- schema assertions ---
    val fields = resDf.schema.fields

    assert(
      fields
        .map(_.name)
        .sameElements(
          Array(
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
        )
    )

    assert(fields(0).dataType == StringType) // RUN_ID
    assert(fields(1).dataType == StringType) // RULE_ID
    assert(fields(2).dataType == StringType) // RULE_TYPE
    assert(fields(3).dataType == StringType) // COLUMN_NAME
    assert(fields(4).dataType == StringType) // STATUS
    assert(fields(5).dataType == StringType) // MESSAGE
    assert(fields(6).dataType == StringType) // METRIC_NAME
    assert(fields(7).dataType == DoubleType) // METRIC_VALUE
    assert(fields(8).dataType == TimestampType) // CREATED_AT

    // --- data assertions ---
    val rows =
      resDf
        .select(
          "RUN_ID",
          "RULE_ID",
          "RULE_TYPE",
          "COLUMN_NAME",
          "STATUS",
          "MESSAGE"
        )
        .as[(String, String, String, String, String, String)]
        .collect()
        .toSeq

    // RULE_TYPE values must NOT be null in actual data
    assert(rows.forall { case (_, _, ruleType, _, _, _) => ruleType != null })

    val rowsSet = rows.toSet

    assert(
      rowsSet.contains(
        ("TEST_RUN", "R1", "COMPLETENESS", "CUSTOMER_ID", "Failure", "msg1")
      )
    )
    assert(
      rowsSet.contains(
        ("TEST_RUN", "R2", "RANGE", "AMOUNT", "Success", "msg2")
      )
    )
  }
}
