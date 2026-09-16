from __future__ import annotations
import json
import os
from pydantic import ValidationError

from .schema import Formulation

SYSTEM_PROMPT = """You are an expert operations research analyst. You will be given a plain-English optimization word problem. Your job is to translate it into a formal mathematical model -- NOT to solve it by hand.

Respond with ONLY a JSON object, no other text, matching exactly this schema:
{
  "sense": "minimize" | "maximize",
  "objective": {"<variable_name>": <coefficient>, ...},
  "variables": [
    {"name": "<variable_name>", "var_type": "continuous" | "integer" | "binary", "low_bound": <number or null>, "up_bound": <number or null>}
  ],
  "constraints": [
    {"expr": {"<variable_name>": <coefficient>, ...}, "sense": "<=" | ">=" | "==", "rhs": <number>, "label": "<short description>"}
  ],
  "claimed_optimal_value": <your best estimate of the optimal objective value, or null if you believe the problem is infeasible>
}

Rules:
- Use short lowercase variable names with no spaces (e.g. "a", "fa", "p1").
- Every variable used in objective/constraints must be declared in "variables".
- Read constraint direction carefully: "at least" is >=, "at most" is <=, "exactly"/"equal to" is ==.
- If the constraints as stated cannot all be satisfied simultaneously, still output your best-attempt model, but set claimed_optimal_value to null and add a variable... no -- just set claimed_optimal_value to null to signal you believe it is infeasible.
- Output ONLY the JSON object. No markdown fences, no explanation.
"""


def _call_groq(problem_text: str) -> str:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=problem_text),
    ])
    return response.content


def _mock_response(problem_id: str) -> str:
    mocks = {
        # correct
        "p01_factory_alloc": {
            "sense": "minimize", "objective": {"a": 4, "b": 6},
            "variables": [
                {"name": "a", "var_type": "continuous", "low_bound": 0, "up_bound": 100},
                {"name": "b", "var_type": "continuous", "low_bound": 0, "up_bound": 80},
            ],
            "constraints": [
                {"expr": {"a": 1}, "sense": "<=", "rhs": 100, "label": "factory A capacity"},
                {"expr": {"b": 1}, "sense": "<=", "rhs": 80, "label": "factory B capacity"},
                {"expr": {"a": 1, "b": 1}, "sense": ">=", "rhs": 120, "label": "minimum demand"},
            ],
            "claimed_optimal_value": 520,
        },
        # deliberately wrong: misreads demand as == instead of >=
        "p02_transport": {
            "sense": "minimize", "objective": {"w1x": 2, "w1y": 3, "w2x": 4, "w2y": 1},
            "variables": [
                {"name": "w1x", "var_type": "continuous", "low_bound": 0},
                {"name": "w1y", "var_type": "continuous", "low_bound": 0},
                {"name": "w2x", "var_type": "continuous", "low_bound": 0},
                {"name": "w2y", "var_type": "continuous", "low_bound": 0},
            ],
            "constraints": [
                {"expr": {"w1x": 1, "w1y": 1}, "sense": "==", "rhs": 50, "label": "warehouse 1 supply (WRONG: should be <=)"},
                {"expr": {"w2x": 1, "w2y": 1}, "sense": "<=", "rhs": 60, "label": "warehouse 2 supply"},
                {"expr": {"w1x": 1, "w2x": 1}, "sense": ">=", "rhs": 40, "label": "store X demand"},
                {"expr": {"w1y": 1, "w2y": 1}, "sense": ">=", "rhs": 55, "label": "store Y demand"},
            ],
            "claimed_optimal_value": 235,
        },
        # deliberately wrong: maximizes (return - cost) instead of return
        "p03_knapsack_budget": {
            "sense": "maximize", "objective": {"p1": 8000, "p2": 4000, "p3": 9000, "p4": 3000},
            "variables": [
                {"name": "p1", "var_type": "binary"}, {"name": "p2", "var_type": "binary"},
                {"name": "p3", "var_type": "binary"}, {"name": "p4", "var_type": "binary"},
            ],
            "constraints": [
                {"expr": {"p1": 20000, "p2": 15000, "p3": 25000, "p4": 10000}, "sense": "<=", "rhs": 50000, "label": "budget"},
            ],
            "claimed_optimal_value": 17000,
        },
        # correct: agent DOES detect infeasibility
        "p04_shift_scheduling": {
            "sense": "minimize", "objective": {"m": 80, "a": 90, "e": 100},
            "variables": [
                {"name": "m", "var_type": "integer", "low_bound": 0},
                {"name": "a", "var_type": "integer", "low_bound": 0},
                {"name": "e", "var_type": "integer", "low_bound": 0},
            ],
            "constraints": [
                {"expr": {"m": 1}, "sense": ">=", "rhs": 2, "label": "morning minimum"},
                {"expr": {"a": 1}, "sense": ">=", "rhs": 3, "label": "afternoon minimum"},
                {"expr": {"e": 1}, "sense": ">=", "rhs": 2, "label": "evening minimum"},
                {"expr": {"m": 1, "a": 1, "e": 1}, "sense": "<=", "rhs": 5, "label": "total workers available"},
            ],
            "claimed_optimal_value": None,
        },
        # correct
        "p05_diet_blend": {
            "sense": "minimize", "objective": {"fa": 3, "fb": 2},
            "variables": [
                {"name": "fa", "var_type": "continuous", "low_bound": 0},
                {"name": "fb", "var_type": "continuous", "low_bound": 0},
            ],
            "constraints": [
                {"expr": {"fa": 2, "fb": 1}, "sense": ">=", "rhs": 10, "label": "protein minimum"},
                {"expr": {"fa": 1, "fb": 2}, "sense": ">=", "rhs": 8, "label": "energy minimum"},
            ],
            "claimed_optimal_value": 12,
        },
    }
    return json.dumps(mocks.get(problem_id, mocks["p05_diet_blend"]))


def formulate(problem_id: str, problem_text: str) -> tuple[Formulation, float | None]:
    if os.environ.get("GROQ_API_KEY"):
        raw = _call_groq(problem_text)
    else:
        raw = _mock_response(problem_id)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Solver agent did not return valid JSON for {problem_id}: {e}\nRaw: {raw[:300]}")

    claimed = data.pop("claimed_optimal_value", None)
    try:
        formulation = Formulation(**data)
    except ValidationError as e:
        raise ValueError(f"Solver agent's JSON didn't match the Formulation schema for {problem_id}: {e}")

    return formulation, claimed
