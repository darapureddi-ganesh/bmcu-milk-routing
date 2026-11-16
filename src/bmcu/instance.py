"""A routing instance: the data both optimisation models read."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FleetConfig


@dataclass
class Instance:
    node_ids: list[int]        # graph node ids; position 0 is the plant
    names: list[str]
    demand: np.ndarray         # litres to collect at each position (0 at the plant)
    ready: np.ndarray          # earliest collection start (min from 05:00)
    deadline: np.ndarray       # latest collection start for milk to stay within its quality window
    service: np.ndarray        # minutes spent at each stop
    dist: np.ndarray           # road km between positions
    time: np.ndarray           # planning travel minutes between positions
    capacity: float            # usable litres per tanker in this plan
    n_vehicles: int
    close: float               # tankers back at the plant by this time
    fixed_cost: float
    cost_per_km: float

    @property
    def n(self) -> int:
        return len(self.node_ids)


def build_instance(nodes: pd.DataFrame, dist: np.ndarray, time: np.ndarray, bmcu_ids: list[int],
                   fleet: FleetConfig = FleetConfig(), capacity_share: float = 1.0,
                   time_factor: float = 1.0, n_vehicles: int | None = None) -> Instance:
    """dist/time are matrices over [plant] + all BMCUs in node order; bmcu_ids picks a subset."""
    all_bmcus = nodes[nodes.kind == "bmcu"].node.tolist()
    pos = [0] + [1 + all_bmcus.index(b) for b in bmcu_ids]
    sub = nodes.set_index("node").loc[[0] + list(bmcu_ids)]
    demand = sub.volume_l.to_numpy(float)
    service = np.where(demand > 0, fleet.service_fixed_min + demand / fleet.pump_l_per_min, 0.0)
    return Instance(
        node_ids=[0] + list(bmcu_ids), names=sub.name.tolist(), demand=demand,
        ready=sub.ready_min.to_numpy(float), deadline=sub.deadline_min.to_numpy(float),
        service=service, dist=dist[np.ix_(pos, pos)], time=time[np.ix_(pos, pos)] * time_factor,
        capacity=fleet.capacity_l * capacity_share,
        n_vehicles=n_vehicles or fleet.n_tankers, close=fleet.plant_close_min,
        fixed_cost=fleet.fixed_cost_inr, cost_per_km=fleet.cost_per_km_inr)


def check_routes(inst: Instance, routes: list[list[int]]) -> dict:
    """Evaluate routes (lists of positions, excluding the plant) with exact float times.

    Returns cost, km and any constraint violations, so both models are judged the same way.
    """
    km, violations, details = 0.0, [], []
    visited = sorted(p for r in routes for p in r)
    if visited != list(range(1, inst.n)):
        violations.append("not every BMCU visited exactly once")
    for r in routes:
        load, t, prev, r_km = 0.0, 0.0, 0, 0.0
        for p in r:
            t += inst.time[prev, p]
            r_km += inst.dist[prev, p]
            t = max(t, inst.ready[p])
            if t > inst.deadline[p] + 1e-6:
                violations.append(f"{inst.names[p]} collected late ({t:.0f} > {inst.deadline[p]:.0f})")
            t += inst.service[p]
            load += inst.demand[p]
            prev = p
        t += inst.time[prev, 0]
        r_km += inst.dist[prev, 0]
        if load > inst.capacity + 1e-6:
            violations.append(f"route over capacity ({load:.0f} L)")
        if t > inst.close + 1e-6:
            violations.append(f"route returns late ({t:.0f})")
        km += r_km
        details.append(dict(stops=len(r), load_l=load, km=r_km, return_min=t))
    cost = inst.fixed_cost * len(routes) + inst.cost_per_km * km
    return dict(cost=cost, km=km, tankers=len(routes), violations=violations, routes=details)
