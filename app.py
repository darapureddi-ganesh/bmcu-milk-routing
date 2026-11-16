"""Interactive front end for the BMCU milk collection model.

Run with:  streamlit run app.py
Every number comes from the same model as run_all.py: OR-Tools routing,
NetworkX road analysis and the SimPy disruption simulation.
"""
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from bmcu import plots  # noqa: E402
from bmcu.config import (SCENARIOS, DisruptionScenario, EconomicsConfig, FleetConfig,  # noqa: E402
                         Policy, ScenarioConfig)
from bmcu.experiment import PlanningError, plan_policies, simulate_all, summarise  # noqa: E402
from bmcu.network import build_graph, closable_crossings, link_criticality, shortest_matrices  # noqa: E402
from bmcu.scenario import generate  # noqa: E402

st.set_page_config(page_title="Milk collection resilience model", layout="wide")

PRESETS = {s.name: s for s in SCENARIOS}
TIME_OPTIONS = list(range(300, 571, 15))  # minutes after 05:00


def clock(minutes: float) -> str:
    m = int(round(minutes))
    return f"{5 + m // 60:02d}:{m % 60:02d}"


def png(draw, *args) -> bytes:
    buf = io.BytesIO()
    draw(*args, buf)
    return buf.getvalue()


# ---------------------------------------------------------------- model calls (cached)
@st.cache_data(show_spinner=False, max_entries=10)
def build_world(n_bmcus, seed, speed):
    nodes = generate(ScenarioConfig(seed=seed, n_bmcus=n_bmcus))
    fleet = FleetConfig(speed_kmph=speed)
    G = build_graph(nodes, fleet)
    bmcus = nodes[nodes.kind == "bmcu"].node.tolist()
    dist, ttime = shortest_matrices(G, [0] + bmcus)
    crit = link_criticality(G, 0, bmcus)
    return nodes, G, dist, ttime, crit


def make_fleet(speed, n_tankers, capacity, close, fixed_cost, per_km):
    return FleetConfig(n_tankers=n_tankers, capacity_l=capacity, speed_kmph=speed,
                       plant_close_min=close, fixed_cost_inr=fixed_cost, cost_per_km_inr=per_km)


def make_policies(buffer_pct, slack_pct):
    share = 1 - buffer_pct / 100
    return (Policy("P0 Cost-optimal", 1.0, 1.0, False),
            Policy("P1 Capacity buffer", share, 1.0, False),
            Policy("P2 Buffer + time slack", share, 1 + slack_pct / 100, False),
            Policy("P3 Buffer + standby tanker", share, 1.0, True))


@st.cache_data(show_spinner=False, max_entries=10)
def plan(world_key, fleet_key, buffer_pct, slack_pct, effort_s):
    nodes, G, dist, ttime, _ = build_world(*world_key)
    fleet = make_fleet(world_key[2], *fleet_key)
    return plan_policies(G, nodes, dist, ttime, fleet, time_limit_s=effort_s,
                         policies=make_policies(buffer_pct, slack_pct))


@st.cache_data(show_spinner=False, max_entries=10)
def simulate(world_key, fleet_key, buffer_pct, slack_pct, effort_s, scen_tuple, days, econ_key):
    nodes, G, *_ = build_world(*world_key)
    plans = plan(world_key, fleet_key, buffer_pct, slack_pct, effort_s)
    fleet = make_fleet(world_key[2], *fleet_key)
    scenarios = tuple(DisruptionScenario(*s) for s in scen_tuple)
    econ = EconomicsConfig(milk_value_inr_per_l=econ_key[0], late_value_loss_share=econ_key[1] / 100)
    runs = simulate_all(G, nodes, plans, n_reps=days, scenarios=scenarios, fleet=fleet, econ=econ)
    return summarise(runs)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    st.caption("Change anything, then press Run.")

    with st.expander("Fleet and plant", expanded=True):
        n_tankers = st.slider("Tankers available", 4, 20, 12)
        capacity = st.slider("Tanker capacity (litres)", 5000, 15000, 10000, step=500)
        speed = st.slider("Average road speed (km/h)", 20, 60, 38)
        close = st.select_slider("Tankers back at the plant by", TIME_OPTIONS, value=450, format_func=clock)

    with st.expander("Disruption to test", expanded=True):
        preset = st.selectbox("Start from", list(PRESETS), index=1)
        p = PRESETS[preset]
        slow = st.slider("Travel slower by (%)", 0, 100, int(round((p.travel_factor - 1) * 100)), key=f"slow{preset}")
        closure = st.slider("Chance each causeway floods (%)", 0, 100, int(round(p.crossing_close_prob * 100)),
                            key=f"close{preset}")
        breakdown = st.slider("Chance a tanker breaks down (%)", 0, 20, int(round(p.breakdown_prob * 100)),
                              key=f"brk{preset}")
        vol = st.slider("Daily milk volume swing (%)", 0, 30, int(round(p.volume_noise_sd * 100)), key=f"vol{preset}")
        noise = st.slider("Travel time unpredictability (%)", 0, 50, int(round(p.travel_noise_sd * 100)),
                          key=f"noise{preset}")
        compare_all = st.checkbox("Also run the three standard scenarios", value=False)

    with st.expander("Planning policies"):
        buffer_pct = st.slider("Capacity buffer: keep this much of each tanker empty (%)", 0, 30, 15)
        slack_pct = st.slider("Time slack: plan as if travel takes longer by (%)", 0, 60, 25)
        st.caption("P0 plans for cost only. P1 adds the capacity buffer. P2 adds time slack. "
                   "P3 adds a standby tanker that collects any milk the plan misses.")

    with st.expander("Costs"):
        fixed_cost = st.number_input("Cost of one tanker for a day (₹)", 500, 20000, 3000, step=500)
        per_km = st.number_input("Cost per km (₹)", 5, 100, 32)
        milk_value = st.number_input("Value of milk (₹ per litre)", 10, 100, 40)
        late_loss = st.slider("Value lost when milk is collected late (%)", 0, 100, 30)

    with st.expander("Collection area and run size"):
        n_bmcus = st.slider("Number of BMCUs", 10, 40, 30)
        seed = st.number_input("Map layout seed", 1, 9999, 42)
        days = st.slider("Simulated days per policy", 20, 500, 100, step=20)
        effort = st.radio("Route planning effort", ["Fast (3 s per plan)", "Thorough (15 s per plan)"])
        effort_s = 3 if effort.startswith("Fast") else 15

    run = st.button("Run model", type="primary", width="stretch")

# ---------------------------------------------------------------- main
st.title("Milk collection under disruption")
st.write("Plans daily tanker routes for village bulk milk cooling units (BMCUs), then tests each plan "
         "against floods, slow roads, volume swings and breakdowns. All data is synthetic.")

world_key = (n_bmcus, int(seed), speed)
fleet_key = (n_tankers, capacity, close, fixed_cost, per_km)
custom = ("Your scenario", 1 + slow / 100, noise / 100, vol / 100, closure / 100, breakdown / 100)
scen_tuple = (custom,) + (tuple((s.name, s.travel_factor, s.travel_noise_sd, s.volume_noise_sd,
                                  s.crossing_close_prob, s.breakdown_prob) for s in SCENARIOS) if compare_all else ())
econ_key = (milk_value, late_loss)

if run:
    st.session_state["args"] = (world_key, fleet_key, buffer_pct, slack_pct, effort_s, scen_tuple, days, econ_key)

if "args" not in st.session_state:
    st.info("Set the sliders on the left and press **Run model**. The first run takes about 30 seconds.")
    st.stop()

args = st.session_state["args"]
if args != (world_key, fleet_key, buffer_pct, slack_pct, effort_s, scen_tuple, days, econ_key):
    st.warning("You've changed settings since the last run. Press Run model to update the results.")

w_key, f_key, b_pct, s_pct, e_s, sc_t, n_days, ec_key = args
try:
    with st.spinner("Building the road network..."):
        nodes, G, dist, ttime, crit = build_world(*w_key)
    with st.spinner("Planning tanker routes for the four policies..."):
        plans = plan(w_key, f_key, b_pct, s_pct, e_s)
    with st.spinner(f"Simulating {n_days} days for each policy..."):
        summary = simulate(w_key, f_key, b_pct, s_pct, e_s, sc_t, n_days, ec_key)
except PlanningError as err:
    st.error(f"No workable plan with these settings. {err} "
             "Try more tankers, bigger tankers, a later return time or a smaller capacity buffer.")
    st.stop()

bm = nodes[nodes.kind == "bmcu"]
base = plans["P0 Cost-optimal"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Milk to collect", f"{bm.volume_l.sum():,.0f} L/day", f"{len(bm)} BMCUs", delta_color="off")
c2.metric("Cheapest plan", f"{base['tankers']} tankers", f"{base['planned_km']:,.0f} km", delta_color="off")
c3.metric("Cheapest plan cost", f"₹{base['planned_cost']:,.0f}/day")
c4.metric("Fullest tanker", f"{max(r['load_l'] for r in base['route_details']):,.0f} L",
          f"of {f_key[1]:,} L", delta_color="off")

tab_res, tab_plan, tab_net, tab_how = st.tabs(["Disruption results", "Plans and routes", "Road network", "How it works"])

with tab_res:
    for scen in dict.fromkeys(summary.scenario):
        d = summary[summary.scenario == scen].copy()
        best = d.loc[d.total_cost_inr.idxmin()]
        st.subheader(scen)
        st.write(f"Lowest expected total cost: **{best.policy}** at ₹{best.total_cost_inr:,.0f} a day, "
                 f"with milk stranded on {best.p_any_stranded:.0%} of days.")
        table = pd.DataFrame({
            "Policy": d.policy,
            "On-time share": (100 * d.service_level).round(1).astype(str) + "%",
            "Milk at risk (L/day)": d.at_risk_l.round(0).astype(int),
            "Days with stranded milk": (100 * d.p_any_stranded).round(0).astype(int).astype(str) + "%",
            "Transport (₹/day)": d.transport_inr.round(0).astype(int),
            "Total cost (₹/day)": d.total_cost_inr.round(0).astype(int),
        })
        st.dataframe(table, hide_index=True, width="stretch")
        chart = pd.DataFrame({"Transport": d.transport_inr.values,
                              "Milk lost or degraded": (d.total_cost_inr - d.transport_inr).values},
                             index=[p.split(" ", 1)[1] for p in d.policy])
        left, right = st.columns(2)
        with left:
            st.caption("Expected cost per day (₹)")
            st.bar_chart(chart, horizontal=True, stack=True, color=["#2a78d6", "#eb6834"])
        with right:
            st.caption("Milk collected late or stranded (litres per day)")
            st.bar_chart(pd.DataFrame({"Litres at risk": d.at_risk_l.values},
                                      index=[p.split(" ", 1)[1] for p in d.policy]),
                         horizontal=True, color="#2a78d6")

with tab_plan:
    rows = []
    for name, pl in plans.items():
        rows.append({"Policy": name, "Tankers": pl["tankers"], "Planned km": round(pl["planned_km"]),
                     "Planned cost (₹/day)": round(pl["planned_cost"]),
                     "Fullest tanker (L)": round(max(r["load_l"] for r in pl["route_details"])),
                     "Last tanker back": clock(max(r["return_min"] for r in pl["route_details"]))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    pick = st.selectbox("Show routes for", list(plans))
    pl = plans[pick]
    buf = io.BytesIO()
    plots.route_map(G, nodes, pl["routes"], buf, f"{pick}: {pl['tankers']} tankers, {pl['planned_km']:.0f} km")
    st.image(buf.getvalue(), width="stretch")

with tab_net:
    st.write(f"{G.number_of_nodes()} places, {G.number_of_edges()} road links, "
             f"{len(closable_crossings(G))} causeways that can flood, and one highway bridge that stays open.")
    st.image(png(plots.criticality_chart, crit), width="stretch")
    top = crit.head(10)[["name", "crossing", "bridge", "betweenness", "extra_km", "cut_off"]].rename(columns={
        "name": "Road link", "crossing": "Causeway", "bridge": "Highway bridge", "betweenness": "Routes using it",
        "extra_km": "Extra km if closed", "cut_off": "BMCUs cut off"})
    top["Extra km if closed"] = top["Extra km if closed"].round(1)
    st.dataframe(top, hide_index=True, width="stretch")

with tab_how:
    st.markdown("""
**1. Road network.** Each place links to its nearest neighbours. Road length is straight-line distance × 1.3.
Links across the river are causeways that can flood; one highway bridge never closes.

**2. Route planning.** Google OR-Tools plans the cheapest set of tanker routes that respects tanker capacity,
each BMCU's milk-quality window and the plant's return time. In the full project it is checked against an exact
MILP, which it matches on every small test case.

**3. Disruption simulation.** Each plan is replayed day by day in SimPy. Causeways flood at random, travel slows,
milk volumes swing and tankers can break down. Milk a tanker can't collect is stranded, unless the standby tanker
(P3) picks it up. Every policy faces the same random days, so the comparison is fair.

**4. Cost.** Transport (tankers, km, overtime, standby retainer) plus milk lost: full value if stranded,
part of its value if collected after its quality window.

Code: [github.com/darapureddi-ganesh/bmcu-milk-routing](https://github.com/darapureddi-ganesh/bmcu-milk-routing)
""")
