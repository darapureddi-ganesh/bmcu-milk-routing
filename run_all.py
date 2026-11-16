"""Run the full pipeline: scenario -> network analysis -> optimisation -> simulation -> results.

Usage:  python run_all.py            (full run, about 3 minutes)
        python run_all.py --quick    (50 replications, short solver limits)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from bmcu import plots  # noqa: E402
from bmcu.config import N_REPLICATIONS, POLICIES, SCENARIOS, FleetConfig, ScenarioConfig  # noqa: E402
from bmcu.experiment import plan_policies, simulate_all, summarise  # noqa: E402
from bmcu.instance import build_instance, check_routes  # noqa: E402
from bmcu.milp import solve_milp  # noqa: E402
from bmcu.network import build_graph, closable_crossings, link_criticality, shortest_matrices  # noqa: E402
from bmcu.routing import solve_routing  # noqa: E402
from bmcu.scenario import generate  # noqa: E402

ROOT = Path(__file__).parent
DATA, OUT, FIG = ROOT / "data", ROOT / "outputs", ROOT / "outputs" / "figures"


def main(quick: bool = False):
    for d in (DATA, OUT, FIG):
        d.mkdir(parents=True, exist_ok=True)
    reps = 50 if quick else N_REPLICATIONS
    t0 = time.perf_counter()

    # 1. Scenario
    nodes = generate(ScenarioConfig())
    nodes.to_csv(DATA / "nodes.csv", index=False)
    bmcus = nodes[nodes.kind == "bmcu"].node.tolist()
    print(f"[1] Scenario: {len(bmcus)} BMCUs, {nodes[nodes.kind == 'bmcu'].volume_l.sum():,.0f} L/day")

    # 2. Network and vulnerability
    G = build_graph(nodes)
    edges = pd.DataFrame([dict(u=u, v=v, **d) for u, v, d in G.edges(data=True)])
    edges.to_csv(DATA / "road_links.csv", index=False)
    dist, ttime = shortest_matrices(G, [0] + bmcus)
    crit = link_criticality(G, 0, bmcus)
    crit.to_csv(OUT / "link_criticality.csv", index=False)
    print(f"[2] Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} links, "
          f"{len(closable_crossings(G))} causeways; top link adds {crit.extra_km.iloc[0]:.0f} km if closed")

    # 3a. Validate the heuristic against the exact MILP on small instances
    rng = np.random.default_rng(7)
    val_rows = []
    for size in (7, 8, 9, 10):
        sub = sorted(rng.choice(bmcus, size, replace=False).tolist())
        inst = build_instance(nodes, dist, ttime, sub, n_vehicles=4)
        exact = solve_milp(inst, time_limit_s=60 if quick else 300)
        heur = solve_routing(inst, time_limit_s=5 if quick else 10)
        e, h = check_routes(inst, exact["routes"]), check_routes(inst, heur["routes"])
        val_rows.append(dict(bmcus=size, milp_status=exact["status"], milp_cost_inr=round(e["cost"]),
                             milp_seconds=round(exact["seconds"], 1), ortools_cost_inr=round(h["cost"]),
                             gap_pct=round(100 * (h["cost"] - e["cost"]) / e["cost"], 2),
                             feasible=not (e["violations"] or h["violations"])))
    val = pd.DataFrame(val_rows)
    val.to_csv(OUT / "milp_vs_ortools.csv", index=False)
    print("[3a] MILP vs OR-Tools\n" + val.to_string(index=False))

    # 3b. Plan each policy on the full network
    plans = plan_policies(G, nodes, dist, ttime, time_limit_s=10 if quick else 30)
    plan_rows = []
    for name, p in plans.items():
        plan_rows.append(dict(policy=name, tankers=p["tankers"], planned_km=round(p["planned_km"]),
                              planned_cost_inr=round(p["planned_cost"]),
                              max_load_l=round(max(r["load_l"] for r in p["route_details"])),
                              latest_return=f"{5 + int(max(r['return_min'] for r in p['route_details']) // 60):02d}:"
                                            f"{int(max(r['return_min'] for r in p['route_details']) % 60):02d}"))
    plan_df = pd.DataFrame(plan_rows)
    plan_df.to_csv(OUT / "plans.csv", index=False)
    json.dump({k: v["routes"] for k, v in plans.items()}, open(OUT / "routes.json", "w"), indent=1)
    print("[3b] Plans\n" + plan_df.to_string(index=False))

    # 4. Simulate every plan in every scenario
    sims = simulate_all(G, nodes, plans, n_reps=reps)
    sims.to_csv(OUT / "simulation_runs.csv", index=False)
    summary = summarise(sims)
    summary.to_csv(OUT / "simulation_summary.csv", index=False)
    print(f"[4] Simulation: {len(sims):,} simulated days\n" + summary.round(3).to_string(index=False))

    # 5. Figures and results page
    base = plans[POLICIES[0].name]
    plots.route_map(G, nodes, base["routes"], FIG / "route_map.png",
                    f"Cost-optimal plan: {base['tankers']} tankers, {base['planned_km']:.0f} km")
    plots.criticality_chart(crit, FIG / "link_criticality.png")
    plots.cost_breakdown_chart(summary, FIG / "cost_by_policy.png")
    plots.risk_chart(summary, FIG / "milk_at_risk.png")
    write_results(nodes, G, crit, val, plan_df, summary, reps)
    print(f"Done in {time.perf_counter() - t0:.0f} s. See outputs/results.md")


def write_results(nodes, G, crit, val, plan_df, summary, reps):
    def table(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
        return "\n".join(lines)

    s = summary.copy()
    s["service_level"] = (100 * s.service_level).round(1).astype(str) + "%"
    s["at_risk_l"] = s.at_risk_l.round(0).astype(int).map("{:,}".format) + " ± " + s.at_risk_ci.round(0).astype(int).astype(str)
    s["p_any_stranded"] = (100 * s.p_any_stranded).round(0).astype(int).astype(str) + "%"
    s["transport_inr"] = s.transport_inr.round(0).astype(int).map("{:,}".format)
    s["total_cost_inr"] = s.total_cost_inr.round(0).astype(int).map("{:,}".format) + " ± " + s.total_cost_ci.round(0).astype(int).map("{:,}".format)
    s = s[["scenario", "policy", "service_level", "at_risk_l", "p_any_stranded", "transport_inr", "total_cost_inr"]]
    s.columns = ["Scenario", "Policy", "On-time share", "Litres at risk/day", "Days with stranded milk",
                 "Transport ₹/day", "Total cost ₹/day"]

    c = crit.head(5)[["name", "crossing", "bridge", "betweenness", "extra_km", "cut_off"]].copy()
    c["extra_km"] = c.extra_km.round(1)
    c["betweenness"] = c.betweenness.round(0).astype(int)

    scen_rows = pd.DataFrame([dict(Scenario=x.name, **{"Travel time ×": x.travel_factor,
                              "Travel noise (sd)": x.travel_noise_sd, "Volume noise (sd)": x.volume_noise_sd,
                              "Causeway closure prob.": x.crossing_close_prob, "Breakdown prob./tanker": x.breakdown_prob})
                              for x in SCENARIOS])
    text = f"""# Results

Generated by `run_all.py`. All inputs are illustrative (see README, Assumptions).

## Network
{G.number_of_nodes()} nodes, {G.number_of_edges()} road links, {len(closable_crossings(G))} causeways that can flood,
1 highway bridge that stays open. Top 5 links by impact if closed:

{table(c)}

`betweenness` = number of plant-to-BMCU shortest paths using the link; `extra_km` = added total
plant-to-BMCU distance if it closes; `cut_off` = BMCUs that become unreachable.

## Exact MILP vs OR-Tools heuristic (small instances, 4 tankers)
{table(val)}

## Plans (planned on a normal day)
{table(plan_df)}

## Disruption scenarios
{table(scen_rows)}

## Simulation results ({reps} simulated days per scenario and policy)
{table(s)}
"""
    (OUT / "results.md").write_text(text)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
