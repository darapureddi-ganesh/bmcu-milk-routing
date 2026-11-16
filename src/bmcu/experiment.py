"""Plan each policy, then test every plan in every disruption scenario."""
import numpy as np
import pandas as pd

from .config import N_REPLICATIONS, POLICIES, SCENARIOS, SIM_SEED, EconomicsConfig, FleetConfig
from .instance import build_instance, check_routes
from .routing import solve_routing
from .simulation import CollectionDay, day_cost


class PlanningError(RuntimeError):
    """No feasible plan exists for the chosen settings."""


def plan_policies(G, nodes, dist, time, fleet=FleetConfig(), time_limit_s=20, policies=POLICIES) -> dict:
    bmcus = nodes[nodes.kind == "bmcu"].node.tolist()
    nominal = build_instance(nodes, dist, time, bmcus, fleet)
    plans = {}
    for pol in policies:
        inst = build_instance(nodes, dist, time, bmcus, fleet,
                              capacity_share=pol.capacity_share, time_factor=pol.planning_time_factor)
        sol = solve_routing(inst, time_limit_s)
        if not sol["routes"]:
            raise PlanningError(f"No feasible plan for {pol.name}: not enough tankers, capacity or time.")
        check = check_routes(nominal, sol["routes"])  # judge every plan on nominal conditions
        if check["violations"]:
            raise PlanningError(f"{pol.name}: {'; '.join(check['violations'][:3])}")
        routes = [[inst.node_ids[p] for p in r] for r in sol["routes"]]
        plans[pol.name] = dict(policy=pol, routes=routes, positions=sol["routes"],
                               planned_cost=check["cost"], planned_km=check["km"],
                               tankers=check["tankers"], route_details=check["routes"])
    return plans


def simulate_all(G, nodes, plans, n_reps=N_REPLICATIONS, seed=SIM_SEED, scenarios=SCENARIOS,
                 fleet=FleetConfig(), econ=EconomicsConfig()) -> pd.DataFrame:
    """Common random numbers: within a replication, every policy faces the same
    road closures and milk volumes, and the same operational random stream."""
    rows = []
    for s_idx, scen in enumerate(scenarios):
        for rep in range(n_reps):
            for name, plan in plans.items():
                env_rng = np.random.default_rng([seed, s_idx, rep])
                ops_rng = np.random.default_rng([seed, s_idx, rep, 1])
                day = CollectionDay(G, nodes, plan["routes"], scen, plan["policy"], env_rng, ops_rng, fleet)
                r = day.run()
                rows.append(dict(scenario=scen.name, policy=name, rep=rep, **r.__dict__,
                                 **day_cost(r, plan["policy"], fleet, econ)))
    df = pd.DataFrame(rows)
    df["at_risk_l"] = df.late_l + df.stranded_l
    df["service_level"] = df.on_time_l / df.total_l
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    def ci(x):
        return 1.96 * x.std(ddof=1) / np.sqrt(len(x))

    g = df.groupby(["scenario", "policy"], sort=False)
    out = g.agg(service_level=("service_level", "mean"),
                at_risk_l=("at_risk_l", "mean"), at_risk_ci=("at_risk_l", ci),
                stranded_l=("stranded_l", "mean"),
                p_any_stranded=("stranded_l", lambda x: (x > 1).mean()),
                transport_inr=("transport_inr", "mean"),
                total_cost_inr=("total_cost_inr", "mean"), total_cost_ci=("total_cost_inr", ci),
                overtime_min=("overtime_min", "mean")).reset_index()
    return out
