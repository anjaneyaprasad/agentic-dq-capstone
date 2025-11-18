package com.anjaneya.dq

import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
  * Utility to load ../config/deequ_rules_resolved.json into Seq[ConstraintSpec]
  */
object ConstraintSpecLoader {

  def loadFromJson(
      spark: SparkSession,
      path: String
  ): Seq[ConstraintSpec] = {

    import spark.implicits._

    val df: DataFrame = spark.read
      .option("multiLine", "true")
      .json(path)

    df.collect().toSeq.map { row =>
      val table          = row.getAs[String]("table")
      val ruleId         = row.getAs[String]("rule_id")
      val severity       = row.getAs[String]("severity")
      val constraintType = row.getAs[String]("constraint_type")
      val column         = row.getAs[String]("column")

      // params is a struct; convert to Map[String, String], skipping null values
      val paramsRowOpt = Option(row.getAs[Row]("params"))
      val params: Map[String, String] = paramsRowOpt
        .map { paramsRow =>
          paramsRow
            .schema
            .fieldNames
            .flatMap { fieldName =>
              val value = paramsRow.getAs[Any](fieldName)
              if (value == null) {
                None
              } else {
                Some(fieldName -> value.toString)
              }
            }
            .toMap
        }
        .getOrElse(Map.empty[String, String])

      ConstraintSpec(
        table = table,
        rule_id = ruleId,
        severity = severity,
        constraint_type = constraintType,
        column = column,
        params = params
      )
    }
  }
}
