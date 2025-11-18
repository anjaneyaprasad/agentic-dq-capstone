package com.anjaneya.dq

import com.amazon.deequ.checks.Check
import com.amazon.deequ.checks.CheckLevel

/**
  * Case class mirroring one entry from config/deequ_rules_resolved.json
  *
  * Example JSON:
  * {
  *   "table": "fact_sales",
  *   "rule_id": "fact_sales_customer_not_null",
  *   "severity": "error",
  *   "constraint_type": "isComplete",
  *   "column": "customer_id",
  *   "params": {}
  * }
  */
case class ConstraintSpec(
  table: String,
  rule_id: String,
  severity: String,
  constraint_type: String,
  column: String,
  params: Map[String, String]
)

object DeequRuleMapper {

  /**
    * Build a Deequ Check from a list of ConstraintSpec for a given table.
    * For now we only support a few constraint types: isComplete, isUnique, isContainedIn.
    * You can extend this later.
    */
  def buildCheckForTable(
      tableName: String,
      specs: Seq[ConstraintSpec]
  ): Check = {

    var check = Check(
      CheckLevel.Error,
      s"DQ checks for $tableName"
    )

    specs.foreach { spec =>
      spec.constraint_type match {

        case "isComplete" =>
          check = check.isComplete(spec.column)

        case "isUnique" =>
          check = check.isUnique(spec.column)

        case "isContainedIn" =>
          // params("allowed_values") should be a comma-separated string
          val allowedRaw = spec.params.getOrElse(
            "allowed_values",
            throw new IllegalArgumentException(s"allowed_values missing for rule ${spec.rule_id}")
          )
          val allowedValues: Seq[String] = allowedRaw
            .split(",")
            .map(_.trim)
            .filter(_.nonEmpty)
            .toSeq

          check = check.isContainedIn(spec.column, allowedValues.toArray)

        // You can add more cases later: isGreaterThanOrEqualTo, foreignKey, etc.
        case other =>
          throw new IllegalArgumentException(s"Unsupported constraint_type: $other")
      }
    }

    check
  }
}
