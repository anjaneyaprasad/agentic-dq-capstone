package com.anjaneya.dq

import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import com.anjaneya.dq.DqCommon.DqRuleRow

class DqBadRowsSpec extends AnyFunSuite with SparkTestSession {

  test(
    "buildBadRowsForCompleteness should capture NULL values and match DQ_BAD_ROWS schema"
  ) {

    import spark.implicits._

    val dataDf = Seq(
      ("C1", "Alice"),
      ("C2", null.asInstanceOf[String]),
      ("C3", "Charlie")
    ).toDF("CUSTOMER_ID", "CUSTOMER_NAME")

    val rule = DqRuleRow(
      ruleId = "R_DIM_CUSTOMERS_NAME_COMPLETENESS",
      ruleType = "COMPLETENESS",
      columnName = Some("CUSTOMER_NAME"),
      level = None,
      threshold = Some(0.99),
      minValue = None,
      maxValue = None,
      allowedValuesJson = None,
      paramsJson = None
    )

    val runId = "RUN_COMPLETENESS"
    val datasetName = "DIM_CUSTOMERS"
    val primaryKeyCol = "CUSTOMER_ID"

    val badRowsDf: DataFrame =
      DqValidationJob.buildBadRowsForCompleteness(
        dataDf,
        datasetName,
        rule,
        runId,
        primaryKeyCol
      )

    val fields = badRowsDf.schema.fields
    assert(
      fields
        .map(_.name)
        .sameElements(
          Array(
            "RUN_ID",
            "DATASET_NAME",
            "RULE_ID",
            "PRIMARY_KEY",
            "ROW_JSON",
            "VIOLATION_MSG",
            "CREATED_AT"
          )
        )
    )

    assert(fields(0).dataType == StringType)
    assert(fields(1).dataType == StringType)
    assert(fields(2).dataType == StringType)
    assert(fields(3).dataType == StringType)
    assert(fields(4).dataType == StringType) // to_json → STRING
    assert(fields(5).dataType == StringType)
    assert(fields(6).dataType == TimestampType)

    val rows = badRowsDf
      .select(
        "RUN_ID",
        "DATASET_NAME",
        "RULE_ID",
        "PRIMARY_KEY",
        "VIOLATION_MSG"
      )
      .as[(String, String, String, String, String)]
      .collect()
      .toSeq

    // should only capture the row with CUSTOMER_ID = C2
    assert(rows.size == 1)
    assert(rows.head._1 == runId)
    assert(rows.head._2 == datasetName)
    assert(rows.head._3 == rule.ruleId)
    assert(rows.head._4 == "C2")
    assert(rows.head._5.contains("CUSTOMER_NAME is NULL"))
  }

  test("buildBadRowsForUniqueness should capture duplicate values") {

    import spark.implicits._

    val dataDf = Seq(
      ("C1", "Alice"),
      ("C1", "Alice_dup"),
      ("C2", "Bob")
    ).toDF("CUSTOMER_ID", "CUSTOMER_NAME")

    val rule = DqRuleRow(
      ruleId = "R_DIM_CUSTOMERS_ID_UNIQUENESS",
      ruleType = "UNIQUENESS",
      columnName = Some("CUSTOMER_ID"),
      level = None,
      threshold = None,
      minValue = None,
      maxValue = None,
      allowedValuesJson = None,
      paramsJson = None
    )

    val runId = "RUN_UNIQUENESS"
    val datasetName = "DIM_CUSTOMERS"
    val primaryKeyCol = "CUSTOMER_ID"

    val badRowsDf =
      DqValidationJob.buildBadRowsForUniqueness(
        dataDf,
        datasetName,
        rule,
        runId,
        primaryKeyCol
      )

    val keys = badRowsDf.select("PRIMARY_KEY").as[String].collect().toSeq

    // We expect only rows where CUSTOMER_ID = C1 to be captured
    assert(keys.nonEmpty)
    assert(keys.forall(_ == "C1"))
  }

  test("buildBadRowsForRange should capture out-of-range values") {
    import spark.implicits._

    val dataDf = Seq(
      ("S1", 50.0),
      ("S2", -10.0),
      ("S3", 5000.0)
    ).toDF("SALE_ID", "AMOUNT")

    val rule = DqRuleRow(
      ruleId = "R_FACT_SALES_AMOUNT_RANGE",
      ruleType = "RANGE",
      columnName = Some("AMOUNT"),
      level = None,
      threshold = None,
      minValue = Some(0.0),
      maxValue = Some(1000.0),
      allowedValuesJson = None,
      paramsJson = None
    )

    val runId = "RUN_RANGE"
    val datasetName = "FACT_SALES"
    val primaryKeyCol = "SALE_ID"

    val badRowsDf =
      DqValidationJob.buildBadRowsForRange(
        dataDf,
        datasetName,
        rule,
        runId,
        primaryKeyCol
      )

    val ids = badRowsDf.select("PRIMARY_KEY").as[String].collect().toSet
    assert(ids == Set("S2", "S3")) // -10 and 5000 are out of [0, 1000]
  }
}
