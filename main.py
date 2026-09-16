"""
Run the full Satya Q1 pipeline: load problems -> solver agent formulates
each -> verifier grades each -> print a formulation/execution accuracy
dashboard.

Usage:
    python main.py                  # mock mode (no API key needed)
    GROQ_API_KEY=... python main.py # real LLM calls via langchain-groq
"""

from __future__ import annotations
import json
from dotenv import load_dotenv

from satya.schema import Problem
from satya.solver_agent import formulate
from satya.verifier import grade

load_dotenv()


def load_problems(path: str = "problems/problems.json") -> list[Problem]:
    with open(path) as f:
        raw = json.load(f)
    return [Problem(**p) for p in raw]


def run():
    problems = load_problems()
    results = []

    print(f"Loaded {len(problems)} problems.\n")

    for problem in problems:
        print(f"--- {problem.id} ---")
        try:
            agent_formulation, claimed = formulate(problem.id, problem.text)
        except Exception as e:
            print(f"  FAILED to get a formulation: {e}\n")
            continue

        result = grade(problem, agent_formulation, claimed)
        results.append(result)

        print(f"  formulation_correct : {result.formulation_correct}")
        print(f"  execution_correct   : {result.execution_correct}")
        print(f"  diagnosis           : {result.diagnosis}")
        print()

    n = len(results)
    if n == 0:
        print("No results.")
        return

    formulation_acc = sum(r.formulation_correct for r in results) / n
    graded_execution = [r for r in results if r.execution_correct is not None]
    execution_acc = (
        sum(r.execution_correct for r in graded_execution) / len(graded_execution)
        if graded_execution else float("nan")
    )

    print("=" * 50)
    print("SATYA Q1 DASHBOARD")
    print("=" * 50)
    print(f"Problems graded          : {n}")
    print(f"Formulation accuracy     : {formulation_acc:.0%}")
    print(f"Execution accuracy       : "
          f"{execution_acc:.0%}" if graded_execution else "Execution accuracy       : n/a")
    print("=" * 50)


if __name__ == "__main__":
    run()
