package com.anjaneya.dq.config

final case class DatasetConfig(
  name: String,
  source: String,
  table_or_path: String,
  primary_keys: Seq[String],
  partitions: Seq[String],
  critical_columns: Seq[String]
)

final case class DatasetsConfig(
  datasets: Seq[DatasetConfig]
)

final case class ConstraintRule(
  `type`: String,                  // e.g. "completeness", "non_negative", "domain", etc.
  column: String,                  // column name (ignored for some types like size_greater_than)
  allowed_values: Seq[String] = Seq.empty, // for "domain"
  level: String = "ERROR",         // "ERROR" or "WARN"
  min: Option[Double] = None,      // for min_value
  max: Option[Double] = None,      // for max_value
  threshold: Option[Double] = None // for completeness_threshold / size_greater_than
)

final case class DatasetConstraintsConfig(
  dataset: String,
  constraints: Seq[ConstraintRule]
)