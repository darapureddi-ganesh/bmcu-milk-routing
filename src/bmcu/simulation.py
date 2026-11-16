"""Discrete-event simulation of one collection day under disruption (SimPy).

The plan (tanker routes) is fixed in the morning. The day then unfolds with:
  - road closures: each river causeway may be closed (the highway bridge stays open)
  - slower, noisier travel times in bad weather
  - actual BMCU volumes that differ from the plan
  - tanker breakdowns part-way through a route

Each tanker follows its planned stops, re-routing on the open roads. Milk that
a tanker cannot collect (no room, unreachable, breakdown) becomes a recovery
request. With a standby tanker, requests are served first-come first-served;
without one, that milk is stranded for the day.
"""
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import simpy

from .config import DisruptionScenario, EconomicsConfig, FleetConfig, Policy
from .network import closable_crossings, shortest_matrices


@dataclass
class DayResult:
    total_l: float
    on_time_l: float
    late_l: float
    stranded_l: float
    km: float
    overtime_min: float
    tankers: int
    standby_used: bool
    closed_crossings: int


class CollectionDay:
    def __init__(self, G: nx.Graph, nodes: pd.DataFrame, plan: list[list[int]],
                 scenario: DisruptionScenario, policy: Policy, env_rng: np.random.Generator,
                 ops_rng: np.random.Generator, fleet: FleetConfig = FleetConfig()):
        self.fleet, self.scenario, self.policy, self.rng = fleet, scenario, policy, ops_rng
        bm = nodes[nodes.kind == "bmcu"].set_index("node")
        self.order = [0] + bm.index.tolist()
        self.pos = {v: i for i, v in enumerate(self.order)}

        # Environment draws (shared across policies for a given replication).
        crossings = closable_crossings(G)
        closed = frozenset(e for e in crossings if env_rng.random() < scenario.crossing_close_prob)
        self.n_closed = len(closed)
        sd = scenario.volume_noise_sd
        noise = env_rng.lognormal(-sd ** 2 / 2, sd, len(bm))
        self.remaining = {b: float(v) * f for (b, v), f in zip(bm.volume_l.items(), noise)}
        self.total = sum(self.remaining.values())
        self.ready, self.deadline = bm.ready_min.to_dict(), bm.deadline_min.to_dict()
        self.dist, self.time = shortest_matrices(G, self.order, closed)

        self.plan = plan
        self.on_time = self.late = 0.0
        self.km = self.overtime = 0.0
        self.standby_used = False

    # -- helpers ---------------------------------------------------------------
    def _leg(self, a: int, b: int):
        i, j = self.pos[a], self.pos[b]
        if not np.isfinite(self.time[i, j]):
            return None, None
        sd = self.scenario.travel_noise_sd
        factor = self.scenario.travel_factor * self.rng.lognormal(-sd ** 2 / 2, sd)
        return self.time[i, j] * factor, self.dist[i, j]

    def _collect(self, b: int, room: float, start: float) -> tuple[float, bool]:
        amount = min(self.remaining[b], room)
        self.remaining[b] -= amount
        in_window = start <= self.deadline[b]
        if in_window:
            self.on_time += amount
        else:
            self.late += amount
        return amount, in_window

    def _service(self, amount: float) -> float:
        return self.fleet.service_fixed_min + amount / self.fleet.pump_l_per_min

    def _drive(self, env, a, b):
        t, d = self._leg(a, b)
        if t is None:
            return False
        self.km += d
        yield env.timeout(t)
        return True

    # -- processes ---------------------------------------------------------------
    def tanker(self, env, route, requests):
        cap = self.fleet.capacity_l
        load, load_on_time, here = 0.0, 0.0, 0
        breakdown_at = (int(self.rng.integers(0, len(route)))
                        if self.rng.random() < self.scenario.breakdown_prob else None)
        for k, b in enumerate(route):
            if breakdown_at == k:
                # Tanker is out of service. Milk on board is transferred later (counted late).
                self.on_time -= load_on_time
                self.late += load_on_time
                for rest in route[k:]:
                    requests.put(rest)
                return
            ok = yield from self._drive(env, here, b)
            if not ok:
                requests.put(b)  # unreachable on today's roads
                continue
            here = b
            yield env.timeout(max(0.0, self.ready[b] - env.now))
            amount, in_window = self._collect(b, cap - load, env.now)
            load += amount
            load_on_time += amount if in_window else 0.0
            yield env.timeout(self._service(amount))
            if self.remaining[b] > 1e-6:
                requests.put(b)  # tanker full: milk left behind
        ok = yield from self._drive(env, here, 0)
        if ok:
            self.overtime += max(0.0, env.now - self.fleet.plant_close_min)

    def standby(self, env, requests):
        cap = self.fleet.capacity_l
        load, here = 0.0, 0
        while True:
            b = yield requests.get()
            if b is None:
                break
            if self.remaining[b] <= 1e-6:
                continue
            self.standby_used = True
            if load >= cap - 1e-6:  # full: unload at the plant first
                ok = yield from self._drive(env, here, 0)
                here, load = 0, 0.0
            ok = yield from self._drive(env, here, b)
            if not ok:
                continue  # unreachable for the standby too: stays stranded
            here = b
            yield env.timeout(max(0.0, self.ready[b] - env.now))
            amount, _ = self._collect(b, cap - load, env.now)
            load += amount
            yield env.timeout(self._service(amount))
            if self.remaining[b] > 1e-6:
                requests.put(b)
        if here != 0:
            ok = yield from self._drive(env, here, 0)
            if ok:
                self.overtime += max(0.0, env.now - self.fleet.plant_close_min)

    def run(self) -> DayResult:
        env = simpy.Environment()
        requests = simpy.Store(env)
        trucks = [env.process(self.tanker(env, r, requests)) for r in self.plan]
        if self.policy.standby_tanker:
            env.process(self.standby(env, requests))

            def closer(env):
                yield simpy.AllOf(env, trucks)
                requests.put(None)
            env.process(closer(env))
        env.run()
        stranded = sum(self.remaining.values())
        return DayResult(total_l=self.total, on_time_l=self.on_time, late_l=self.late,
                         stranded_l=stranded, km=self.km, overtime_min=self.overtime,
                         tankers=len(self.plan), standby_used=self.standby_used,
                         closed_crossings=self.n_closed)


def day_cost(r: DayResult, policy: Policy, fleet: FleetConfig = FleetConfig(),
             econ: EconomicsConfig = EconomicsConfig()) -> dict:
    transport = (fleet.fixed_cost_inr * r.tankers + fleet.cost_per_km_inr * r.km
                 + econ.overtime_inr_per_min * r.overtime_min
                 + (econ.standby_fixed_inr if policy.standby_tanker else 0.0))
    milk_loss = (econ.milk_value_inr_per_l * r.stranded_l
                 + econ.milk_value_inr_per_l * econ.late_value_loss_share * r.late_l)
    return dict(transport_inr=transport, milk_loss_inr=milk_loss, total_cost_inr=transport + milk_loss)
