from __future__ import annotations
import pulp

from .schema import Formulation, Problem, GradeResult, Sense, VarType


class InfeasibleError(Exception):
    pass


def _build_lp(formulation: Formulation) -> pulp.LpProblem:
    sense = pulp.LpMinimize if formulation.sense == Sense.MINIMIZE else pulp.LpMaximize
    prob = pulp.LpProblem("satya_problem", sense)

    pulp_vars = {}
    for v in formulation.variables:
        if v.var_type == VarType.BINARY:
            cat = "Binary"
        elif v.var_type == VarType.INTEGER:
            cat = "Integer"
        else:
            cat = "Continuous"
        pulp_vars[v.name] = pulp.LpVariable(
            v.name, lowBound=v.low_bound, upBound=v.up_bound, cat=cat
        )

    # objective
    prob += pulp.lpSum(
        coef * pulp_vars[name] for name, coef in formulation.objective.items() if name in pulp_vars
    )

    # constraints
    for c in formulation.constraints:
        lhs = pulp.lpSum(
            coef * pulp_vars[name] for name, coef in c.expr.items() if name in pulp_vars
        )
        if c.sense == "<=":
            prob += lhs <= c.rhs, c.label
        elif c.sense == ">=":
            prob += lhs >= c.rhs, c.label
        else:
            prob += lhs == c.rhs, c.label

    return prob


def solve(formulation: Formulation) -> float | None:
    """returns the true optimal objective value yooo"""
    prob = _build_lp(formulation)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return None
    return pulp.value(prob.objective)


def grade(problem: Problem, agent_formulation: Formulation, agent_claimed_optimum: float | None) -> GradeResult:
    true_optimum = solve(problem.ground_truth)
    agent_solved_optimum = None
    diagnosis_parts = []

    if true_optimum is None:
        try:
            agent_solved_optimum = solve(agent_formulation)
        except Exception:
            agent_solved_optimum = None

        agent_reported_infeasible = agent_claimed_optimum is None and agent_solved_optimum is None
        formulation_correct = agent_reported_infeasible
        diagnosis = (
            "Ground truth is INFEASIBLE. Agent correctly detected infeasibility."
            if formulation_correct else
            "Ground truth is INFEASIBLE, but the agent proposed a formulation that "
            "solves anyway -- likely dropped or mis-stated a constraint."
        )
        return GradeResult(
            problem_id=problem.id,
            formulation_correct=formulation_correct,
            execution_correct=None,
            true_optimum=float("nan"),
            agent_claimed_optimum=agent_claimed_optimum,
            solved_optimum_from_agent_formulation=agent_solved_optimum,
            diagnosis=diagnosis,
        )

    try:
        agent_solved_optimum = solve(agent_formulation)
    except Exception as e:
        return GradeResult(
            problem_id=problem.id,
            formulation_correct=False,
            execution_correct=None,
            true_optimum=true_optimum,
            agent_claimed_optimum=agent_claimed_optimum,
            solved_optimum_from_agent_formulation=None,
            diagnosis=f"Agent formulation could not even be built/solved: {e}",
        )

    TOL = 1e-3
    if agent_solved_optimum is None:
        formulation_correct = False
        diagnosis_parts.append("Agent's formulation is infeasible/unbounded but ground truth is feasible.")
    elif abs(agent_solved_optimum - true_optimum) < TOL:
        formulation_correct = True
        diagnosis_parts.append("Agent's formulation, when solved exactly, matches the true optimum.")
    else:
        formulation_correct = False
        diagnosis_parts.append(
            f"Agent's formulation solves to {agent_solved_optimum:.2f}, "
            f"true optimum is {true_optimum:.2f} -- formulation differs from ground truth "
            f"(wrong objective, wrong/missing constraint, or wrong constraint direction)."
        )

    execution_correct = None
    if formulation_correct and agent_claimed_optimum is not None:
        execution_correct = abs(agent_claimed_optimum - agent_solved_optimum) < TOL
        if not execution_correct:
            diagnosis_parts.append(
                f"Formulation was right, but the agent's OWN claimed answer "
                f"({agent_claimed_optimum:.2f}) doesn't match what its formulation "
                f"actually solves to ({agent_solved_optimum:.2f}) -- execution/arithmetic error."
            )

    return GradeResult(
        problem_id=problem.id,
        formulation_correct=formulation_correct,
        execution_correct=execution_correct,
        true_optimum=true_optimum,
        agent_claimed_optimum=agent_claimed_optimum,
        solved_optimum_from_agent_formulation=agent_solved_optimum,
        diagnosis=" ".join(diagnosis_parts),
    )
