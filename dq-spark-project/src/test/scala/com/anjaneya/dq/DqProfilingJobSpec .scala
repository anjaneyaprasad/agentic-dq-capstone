package com.anjaneya.dq

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.SparkSession

import com.amazon.deequ.analyzers._

class DqProfilingJobSpec extends AnyFunSuite {

  // tiny local Spark just for tests, no Snowflake used here
  private val spark: SparkSession =
    SparkSession
      .builder()
      .appName("dq-profiling-job-spec")
      .master("local[*]")
      .getOrCreate()

  import spark.implicits._

  test("buildProfilingAnalyzers adds numeric stats only for numeric columns") {
    val df = Seq(
      (1, 10.5, "a"),
      (2, 20.0, "b")
    ).toDF("id_int", "amount_double", "name_str")

    val analyzers = DqProfilingJob.buildProfilingAnalyzers(df)

    // Collect which columns each analyzer applies to
    val completenessCols = analyzers.collect { case c: Completeness =>
      c.column
    }.toSet

    val approxCountDistinctCols = analyzers.collect {
      case a: ApproxCountDistinct => a.column
    }.toSet

    val dataTypeCols = analyzers.collect {
      case d: com.amazon.deequ.analyzers.DataType => d.column
    }.toSet

    val minCols = analyzers.collect { case m: Minimum =>
      m.column
    }.toSet

    val maxCols = analyzers.collect { case m: Maximum =>
      m.column
    }.toSet

    val meanCols = analyzers.collect { case m: Mean =>
      m.column
    }.toSet

    // --- Assertions ---

    // 1) Completeness for ALL columns
    assert(completenessCols == Set("id_int", "amount_double", "name_str"))

    // 2) ApproxCountDistinct for ALL columns
    assert(
      approxCountDistinctCols == Set("id_int", "amount_double", "name_str")
    )

    // 3) DataType analyzer for ALL columns
    assert(dataTypeCols == Set("id_int", "amount_double", "name_str"))

    // 4) Numeric stats ONLY for numeric columns
    assert(minCols == Set("id_int", "amount_double"))
    assert(maxCols == Set("id_int", "amount_double"))
    assert(meanCols == Set("id_int", "amount_double"))
    assert(!minCols.contains("name_str"))
    assert(!maxCols.contains("name_str"))
    assert(!meanCols.contains("name_str"))
  }

  test("datasetsFromMetaDf picks only active datasets") {
    val df = Seq(
      ("FACT_SALES", true),
      ("DIM_CUSTOMERS", true),
      ("DIM_OLD", false)
    ).toDF("DATASET_NAME", "IS_ACTIVE")

    val result = DqProfilingJob.datasetsFromMetaDf(df)

    assert(result == Seq("DIM_CUSTOMERS", "FACT_SALES"))
  }
}
