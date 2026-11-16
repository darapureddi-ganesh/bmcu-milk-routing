"""Exact mixed-integer model of the milk collection problem (CVRPTW).

Two-index formulation:
  x[i,j] = 1 if a tanker drives directly from stop i to stop j
  t[i]   = time collection starts at BMCU i
  q[i]   = litres on board after collecting at BMCU i

minimise   fixed cost x tankers used  +  cost per km x km driven
subject to
  every BMCU is entered once and left once
  at most K tankers leave the plant, and as many return
  time:     t[j] >= t[i] + service[i] + travel[i,j]  if x[i,j] = 1      (big-M)
  windows:  ready[i] <= t[i] <= deadline[i]   (waiting allowed)
  return:   t[i] + service[i] + travel[i,0] <= plant close  if x[i,0] = 1
  load:     q[j] >= q[i] + demand[j]          if x[i,j] = 1     (also removes sub-tours)
            demand[i] <= q[i] <= capacity

Solved exactly with HiGHS. Practical for small instances; the full network
uses the OR-Tools heuristic in routing.py, checked against this model.
"""
import time as _time

import pulp

from .instance import Instance


def solve_milp(inst: Instance, time_limit_s: int = 300) -> dict:
    N = range(inst.n)
    C = range(1, inst.n)
    arcs = [(i, j) for i in N for j in N if i != j]
    M = inst.close + inst.service.max() + inst.time.max()

    m = pulp.LpProblem("milk_collection_cvrptw", pulp.LpMinimize)
    x = {(i, j): m.add_variable(f"x_{i}_{j}", cat="Binary") for i, j in arcs}
    t = {i: m.add_variable(f"t_{i}", float(inst.ready[i]), float(inst.deadline[i])) for i in C}
    q = {i: m.add_variable(f"q_{i}", float(inst.demand[i]), float(inst.capacity)) for i in C}

    m += (inst.fixed_cost * pulp.lpSum(x[0, j] for j in C)
          + inst.cost_per_km * pulp.lpSum(inst.dist[i, j] * x[i, j] for i, j in arcs))

    for i in C:
        m += pulp.lpSum(x[j, i] for j in N if j != i) == 1, f"enter_{i}"
        m += pulp.lpSum(x[i, j] for j in N if j != i) == 1, f"leave_{i}"
    m += pulp.lpSum(x[0, j] for j in C) <= inst.n_vehicles, "fleet"
    m += pulp.lpSum(x[0, j] for j in C) == pulp.lpSum(x[j, 0] for j in C), "return_all"

    for j in C:
        m += t[j] >= inst.time[0, j] - M * (1 - x[0, j]), f"first_{j}"
    for i in C:
        for j in C:
            if i != j:
                m += t[j] >= t[i] + inst.service[i] + inst.time[i, j] - M * (1 - x[i, j]), f"time_{i}_{j}"
                m += q[j] >= q[i] + inst.demand[j] - inst.capacity * (1 - x[i, j]), f"load_{i}_{j}"
        m += t[i] + inst.service[i] + inst.time[i, 0] <= inst.close + M * (1 - x[i, 0]), f"back_{i}"

    start = _time.perf_counter()
    stats = m.solve(pulp.HiGHS(msg=False, timeLimit=time_limit_s))
    secs = _time.perf_counter() - start

    status = stats.status.name  # "Optimal", "TimeLimit", "Infeasible", ...
    routes = []
    if status in ("Optimal", "TimeLimit"):
        succ = {(i): j for (i, j) in arcs if (x[i, j].value() or 0) > 0.5}
        for j in C:
            if (x[0, j].value() or 0) > 0.5:
                route, cur = [], j
                while cur != 0:
                    route.append(cur)
                    cur = succ[cur]
                routes.append(route)
    objective = m.objective.value() if routes else None
    return dict(status=status, objective=objective, best_bound=getattr(stats, "best_bound", None),
                seconds=secs, routes=routes)
