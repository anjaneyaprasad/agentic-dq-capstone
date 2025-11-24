# python-services/nl_constraints_graph/models.py
from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

# Define allowed rule types and levels 
RuleType = Literal[
    "completeness",
    "completeness_threshold",
    "non_negative",
    "domain",
    "unique",
    "size_greater_than",
    "min_value",
    "max_value",
]

# Define allowed severity levels for rules
LevelType = Literal["ERROR", "WARN"]

# Data model for a single rule specification
# e.g., completeness of column "age" must be >= 0.9
# represented as a RuleSpec instance with type="completeness_threshold", column="age", threshold=0.9
class RuleSpec(BaseModel):
    dataset: str
    type: RuleType
    column: str = ""
    level: LevelType = "ERROR"
    threshold: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    allowed_values: List[str] = Field(default_factory=list)

    # Convert to dict format suitable for YAML output in Deequ-like style
    # e.g., {"type": "completeness_threshold", "column": "age", "threshold": 0.9, "level": "ERROR"}
    def to_yaml_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": self.type,
            "column": self.column,
            "level": self.level,
        }
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.allowed_values:
            d["allowed_values"] = self.allowed_values
        return d

# Data model for the NL request
# e.g., dataset="users", prompt="Ensure age is non-negative and completeness >= 0.9"
# represented as NLRequest instance
class NLRequest(BaseModel):
    dataset: str
    prompt: str
    apply: bool = False   # whether to actually write YAML

# Data model for the overall graph state
# flows through the LangGraph workflow
# contains the request, inferred rules, validation status, messages, etc. 
class GraphState(BaseModel):
    """
    State object that flows through LangGraph.
    """
    request: NLRequest
    columns: List[str] = Field(default_factory=list)
    inferred_rules: List[RuleSpec] = Field(default_factory=list)
    validation_ok: bool = True
    validation_messages: List[str] = Field(default_factory=list)
    merged_yaml: Optional[Dict[str, Any]] = None
    yaml_path: Optional[str] = None
    anomaly_messages: List[str] = []
    user_feedback: Optional[str] = None
    refinement_attempts: int = 0
    max_refinements: int = 2
    self_healing_enabled: bool = False