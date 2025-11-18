package com.anjaneya.dq

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row

import com.amazon.deequ.VerificationSuite
import com.amazon.deequ.checks.{Check, CheckLevel}
import com.amazon.deequ.constraints.ConstraintStatus

class RuleMappingSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder()
      .appName("RuleMappingSpecTest")
      .master("local[*]")
      .getOrCreate()

    // Keep logs quieter
    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
    }
    super.afterAll()
  }

  private def sampleFactSalesDF(): DataFrame = {
    // Small dataset:
    // - customer_id is always non-null  => completeness check should PASS
    // - payment_method includes "CRYPTO" => in_set check should FAIL
    val schema = StructType(Seq(
      StructField("sale_id", IntegerType, nullable = false),
      StructField("customer_id", IntegerType, nullable = false),
      StructField("payment_method", StringType, nullable = true)
    ))

    val rows = Seq(
      Row(1, 1001, "CARD"),
      Row(2, 1002, "UPI"),
      Row(3, 1003, "CRYPTO") // out-of-domain on purpose
    )

    spark.createDataFrame(spark.sparkContext.parallelize(rows), schema)
  }

  test("maps JSON-like specs into a Deequ Check and produces expected constraint results") {
    // --- Arrange: two rules as if they came from JSON exporter ---
    val specs = Seq(
      ConstraintSpec(
        table = "fact_sales",
        rule_id = "fact_sales_customer_not_null",
        severity = "error",
        constraint_type = "isComplete",
        column = "customer_id",
        params = Map.empty
      ),
      ConstraintSpec(
        table = "fact_sales",
        rule_id = "fact_sales_payment_method_in_set",
        severity = "warning",
        constraint_type = "isContainedIn",
        column = "payment_method",
        params = Map("allowed_values" -> "CARD,UPI,CASH,WALLET")
      )
    )

    val df = sampleFactSalesDF()

    // --- Act: build a Deequ Check from the specs and run it ---
    val check: Check = DeequRuleMapper.buildCheckForTable("fact_sales", specs)

    val verificationResult = VerificationSuite()
      .onData(df)
      .addChecks(Seq(check))
      .run()

    // There's only one Check instance, so we can grab its result directly
    val checkResult = verificationResult.checkResults.values.head

    val constraintResults = checkResult.constraintResults

    // --- Assert: completeness on customer_id is SUCCESS ---
    val customerCompletenessStatusOpt = constraintResults.find { cr =>
      cr.constraint.toString.contains("CompletenessConstraint") &&
      cr.constraint.toString.contains("customer_id")
    }.map(_.status)

    assert(
      customerCompletenessStatusOpt.contains(ConstraintStatus.Success),
      s"Expected CompletenessConstraint on customer_id to be Success, got: $customerCompletenessStatusOpt"
    )

    // --- Assert: in_set on payment_method is FAILURE (due to CRYPTO) ---
    val paymentInSetStatusOpt = constraintResults.find { cr =>
      cr.constraint.toString.contains("ContainedIn") ||  // wording differs slightly by version
      (cr.constraint.toString.contains("payment_method") &&
       cr.constraint.toString.contains("CARD") &&
       cr.constraint.toString.contains("WALLET"))
    }.map(_.status)

    assert(
      paymentInSetStatusOpt.contains(ConstraintStatus.Failure),
      s"Expected in_set constraint on payment_method to be Failure (because of CRYPTO), got: $paymentInSetStatusOpt"
    )

    // Sanity: check metadata
    assert(check.level == CheckLevel.Error)
    assert(check.description.contains("fact_sales"))
  }
}

