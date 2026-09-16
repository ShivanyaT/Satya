"""
Core data model for Satya.

A Problem is hand-authored ground truth: the plain-English text an LLM sees,
plus the TRUE formulation (used only by the verifier, never shown to the
solver agent) and the true optimal value (used as a final sanity check).

A Formulation is what the Solver Agent produces from the problem text —
this is the thing that actually gets graded.
"""

from __future__ import annotations
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class Sense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class VarType(str, Enum):
    CONTINUOUS = "continuous"   # LP
    INTEGER = "integer"         # ILP
    BINARY = "binary"           # 0/1 ILP


class Variable(BaseModel):
    name: str
    var_type: VarType = VarType.CONTINUOUS
    low_bound: float | None = 0.0
    up_bound: float | None = None


class Constraint(BaseModel):
    # e.g. {"x1": 2, "x2": 3} <= 100  ->  exp={"x1": 2, "x2": 3}, sense="<=", rhs=100
    expr: dict[str, float]
    sense: Literal["<=", ">=", "=="]
    rhs: float
    # optional human-readable label, useful for diagnosing WHICH constraint
    # the model got wrong, not just that something was wrong
    label: str | None = None


class Formulation(BaseModel):
    sense: Sense
    objective: dict[str, float]          # e.g. {"x1": 5, "x2": 4}
    variables: list[Variable]
    constraints: list[Constraint]
    claimed_optimal_value: float | None = None  # the agent's own claimed answer, if it computed one


class Problem(BaseModel):
    id: str
    text: str                              # plain-English problem statement (goes to the LLM)
    family: Literal["LP", "ILP"] = "LP"
    ground_truth: Formulation              # NEVER shown to the solver agent
    notes: str | None = None               # e.g. "the non-obvious binding constraint is..."


class GradeResult(BaseModel):
    problem_id: str
    formulation_correct: bool
    execution_correct: bool | None   # None if formulation was wrong (can't grade execution on a broken setup)
    true_optimum: float
    agent_claimed_optimum: float | None
    solved_optimum_from_agent_formulation: float | None
    diagnosis: str
