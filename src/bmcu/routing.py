"""Heuristic routing for the full network with Google OR-Tools.

Same problem as milp.py (capacity, quality windows, plant closing time,
fixed + per-km cost), solved with guided local search because the exact
model does not scale to 30 BMCUs in practical time.
"""
import math

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .instance import Instance


def solve_routing(inst: Instance, time_limit_s: int = 20) -> dict:
    manager = pywrapcp.RoutingIndexManager(inst.n, inst.n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Cost in rupees (integer). Fixed cost is charged per tanker used.
    def cost_cb(a, b):
        i, j = manager.IndexToNode(a), manager.IndexToNode(b)
        return int(round(inst.cost_per_km * inst.dist[i, j]))

    cost_idx = routing.RegisterTransitCallback(cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)
    routing.SetFixedCostOfAllVehicles(int(inst.fixed_cost))

    def load_cb(a):
        return int(math.ceil(inst.demand[manager.IndexToNode(a)]))

    load_idx = routing.RegisterUnaryTransitCallback(load_cb)
    routing.AddDimensionWithVehicleCapacity(load_idx, 0, [int(inst.capacity)] * inst.n_vehicles, True, "Load")

    # Time in whole minutes, rounded up so OR-Tools never under-estimates.
    def time_cb(a, b):
        i, j = manager.IndexToNode(a), manager.IndexToNode(b)
        return int(math.ceil(inst.service[i] + inst.time[i, j]))

    time_idx = routing.RegisterTransitCallback(time_cb)
    horizon = int(inst.close)
    routing.AddDimension(time_idx, horizon, horizon, False, "Time")
    tdim = routing.GetDimensionOrDie("Time")
    for p in range(1, inst.n):
        tdim.CumulVar(manager.NodeToIndex(p)).SetRange(int(math.ceil(inst.ready[p])), int(inst.deadline[p]))
    for v in range(inst.n_vehicles):
        tdim.CumulVar(routing.End(v)).SetRange(0, horizon)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_s

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return dict(status="infeasible", objective=None, routes=[])
    routes = []
    for v in range(inst.n_vehicles):
        idx, route = routing.Start(v), []
        idx = sol.Value(routing.NextVar(idx))
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        if route:
            routes.append(route)
    return dict(status="feasible", objective=sol.ObjectiveValue(), routes=routes)
