from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class Severity(str, Enum):
    error = "error"
    warning = "warning"
    info = "info"


class CheckType(str, Enum):
    not_null = "not_null"
    unique = "unique"
    greater_or_equal = "greater_or_equal"
    less_or_equal = "less_or_equal"
    in_set = "in_set"


class DQRule(BaseModel):
    """
    One column-level data quality rule.
    Examples:
      - NOT NULL
      - UNIQUE
      - quantity >= 0
      - status in ('SUCCESS', 'FAILED')
    """

    id: str = Field(..., description="Unique rule id within the table")
    column: str = Field(..., description="Target column name")
    check: CheckType = Field(..., description="Type of check")
    severity: Severity = Field(default=Severity.warning)
    threshold: Optional[float] = Field(
        default=None,
        description="Numeric threshold for greater_or_equal / less_or_equal",
    )
    allowed_values: Optional[List[str]] = Field(
        default=None,
        description="Allowed values for in_set checks",
    )

    @validator("threshold", always=True)
    def validate_threshold(cls, value, values):
        check = values.get("check")
        if check in {CheckType.greater_or_equal, CheckType.less_or_equal} and value is None:
            raise ValueError(f"threshold is required for check='{check.value}'")
        return value

    @validator("allowed_values", always=True)
    def validate_allowed_values(cls, value, values):
        check = values.get("check")
        if check == CheckType.in_set and (not value or len(value) == 0):
            raise ValueError("allowed_values is required and must be non-empty for check='in_set'")
        return value


class ForeignKeyRule(BaseModel):
    """
    Represents logical FK checks to be implemented in Spark/Deequ:
    fact_sales.customer_id -> dim_customers.customer_id
    """
    id: str
    local_column: str
    referenced_table: str
    referenced_column: str
    severity: Severity = Severity.error


class TableConfig(BaseModel):
    name: str
    criticality: Optional[str] = Field(
        default="medium", description="Business criticality: low, medium, high, critical"
    )
    rules: List[DQRule] = Field(default_factory=list)
    foreign_keys: List[ForeignKeyRule] = Field(default_factory=list)


class DQConfig(BaseModel):
    version: int = 1
    default_severity: Severity = Severity.warning
    tables: List[TableConfig]

    def get_table(self, name: str) -> Optional[TableConfig]:
        for t in self.tables:
            if t.name == name:
                return t
        return None
