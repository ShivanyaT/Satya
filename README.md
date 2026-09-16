# Satya — Q1 Starter

Working scaffold: Solver Agent -> Verifier -> formulation/execution accuracy
dashboard, exactly as scoped in the Q1 plan. Runs offline with mock data out
of the box; swap in your Groq key when ready.

You should see 5 problems graded, a per-problem diagnosis, and a formulation/
execution accuracy dashboard at the bottom. Two of the five mock formulations
are deliberately wrong on purpose (see `satya/solver_agent.py`) — that's
intentional, so your first run actually demonstrates the grading catching
something instead of a suspicious 100%.

## What's here

```
satya/
  schema.py        # Problem / Formulation / GradeResult data model
  solver_agent.py   # LLM prompt + JSON parsing (mock + real Groq modes)
  verifier.py        # Builds & solves the LP/ILP with PuLP, grades it
problems/
  problems.json     # 5 hand-authored problems w/ ground truth
main.py              # runs the pipeline end to end, prints the dashboard
```

## Next steps, in order

1. **Get a real Groq run working.** Drop in your API key, run it, and look
   at where the *real* model actually fails — formulation or execution.
   That first real dashboard is worth writing down; it's your first genuine
   data point.
2. **Grow the problem set to ~20–30.** Follow the pattern in
   `problems.json`: plain text, a `ground_truth` Formulation, and a `notes`
   field explaining the non-obvious trap (if any). Keep mixing in a couple
   of "boring, no trick" problems (like p05) as controls.
3. **Add the template-variation generator now, not in Q2.** For each
   hand-written problem, write a small function that re-rolls the numeric
   coefficients/RHS values and re-derives the new ground truth
   programmatically (since these are linear, the solver can just re-solve
   the varied version — you don't have to hand-recompute). This is how you
   get from 25 problems to a few hundred without writing them by hand.
4. **Run the baseline sweep (Q2).** Point `solver_agent.py` at 2-3 other
   models (different sizes/providers — `langchain` makes swapping the LLM
   client a small change) and compare formulation vs. execution accuracy
   across them. This is your first real benchmark table.
5. **Log every graded result to disk**, not just stdout — you'll want the
   full history once you start collecting the verified-trace dataset for
   fine-tuning in Q3. A single append-only `results.jsonl` (one `GradeResult`
   per line) is enough for now; no database needed yet.

## Design notes worth knowing before you extend this

- The Verifier never asks an LLM whether something is right — every grade
  comes from actually solving the math with PuLP/CBC. Keep it that way; the
  moment an LLM grades another LLM, the "verifiable" in verifiable-reward
  is gone.
- `execution_correct` is only meaningful right now on problems where the
  agent both formulates *and* claims a numeric answer — it's checking
  whether the agent's own arithmetic matches its own formulation, not
  whether it matches the true optimum. That's correct by design (see the
  docstring in `verifier.py`), but easy to misread at 1am, so it's called
  out here twice.
- Infeasible ground-truth problems (like p04) are a deliberate part of the
  set, not a bug — a system that always outputs *some* confident number is
  exactly the failure mode Satya exists to catch.
