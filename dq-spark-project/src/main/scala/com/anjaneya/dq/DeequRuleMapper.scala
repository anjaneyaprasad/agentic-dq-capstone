package com.anjaneya.dq

import com.amazon.deequ.checks.Check
import com.amazon.deequ.checks.CheckLevel

case class ConstraintSpec(
  table: String,
  rule_id: String,
  severity: String,
  constraint_type: String,
  column: String,
  params: Map[String, String]
)

object DeequRuleMapper {

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

        case "isGreaterThanOrEqualTo" =>
          val thrStr = spec.params.getOrElse(
            "threshold",
            throw new IllegalArgumentException(s"threshold missing for rule ${spec.rule_id}")
          )
          val thr = thrStr.toDouble
          // Deequ: enforce min(column) >= threshold
          check = check.hasMin(
            spec.column,
            _ >= thr,
            Some(s"${spec.column} >= $thr")
          )

        case "isLessThanOrEqualTo" =>
          val thrStr = spec.params.getOrElse(
            "threshold",
            throw new IllegalArgumentException(s"threshold missing for rule ${spec.rule_id}")
          )
          val thr = thrStr.toDouble
          // Deequ: enforce max(column) <= threshold
          check = check.hasMax(
            spec.column,
            _ <= thr,
            Some(s"${spec.column} <= $thr")
          )

        case "foreignKey" =>
          // For now, foreign keys are validated via separate join logic,
          // not as a direct Deequ Check constraint. No-op here.
          ()

        // You can add more cases (foreignKey, etc.) later
        case other =>
          throw new IllegalArgumentException(s"Unsupported constraint_type: $other")
      }
    }

    check
  }
}
