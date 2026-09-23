# Arenix Coverage Optimizer v1

The optimizer is intentionally separate from sport prediction models. It consumes calibrated categorical probabilities and produces ticket portfolios for configurable budgets such as 16, 32, 64 and 128 lines.

Inputs are event-specific labels and probabilities, so the same engine can support Protouch L/D/V or other multisport pool formats without assuming NFL-specific variables.

The v1 objective balances estimated joint probability and line diversity. It uses beam search to avoid exhaustive Cartesian enumeration. Joint line probability currently assumes event independence; correlation is explicitly reserved for a later layer rather than hidden in the optimizer.

Historical evaluation must compare each budget using preregistered/OOS probabilities and report best hits, count of lines at prize thresholds, distance to the realized result, coverage and cost. The optimizer must never use realized outcomes when constructing lines.
