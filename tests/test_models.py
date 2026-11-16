"""Sanity tests. Run with:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from bmcu.config import POLICIES, SCENARIOS  # noqa: E402
from bmcu.instance import build_instance, check_routes  # noqa: E402
from bmcu.milp import solve_milp  # noqa: E402
from bmcu.network import build_graph, shortest_matrices  # noqa: E402
from bmcu.routing import solve_routing  # noqa: E402
from bmcu.scenario import generate  # noqa: E402
from bmcu.simulation import CollectionDay  # noqa: E402


@pytest.fixture(scope="module")
def world():
    nodes = generate()
    G = build_graph(nodes)
    bmcus = nodes[nodes.kind == "bmcu"].node.tolist()
    dist, time = shortest_matrices(G, [0] + bmcus)
    return nodes, G, bmcus, dist, time


def test_network_connected(world):
    nodes, G, bmcus, dist, _ = world
    assert np.isfinite(dist).all()


def test_heuristic_matches_exact_on_small_instance(world):
    nodes, G, bmcus, dist, time = world
    inst = build_instance(nodes, dist, time, bmcus[:7], n_vehicles=4)
    exact = check_routes(inst, solve_milp(inst, 60)["routes"])
    heur = check_routes(inst, solve_routing(inst, 5)["routes"])
    assert not exact["violations"] and not heur["violations"]
    assert heur["cost"] <= exact["cost"] * 1.01


def test_milk_is_conserved_in_simulation(world):
    nodes, G, bmcus, dist, time = world
    inst = build_instance(nodes, dist, time, bmcus)
    routes = [[inst.node_ids[p] for p in r] for r in solve_routing(inst, 5)["routes"]]
    for scen in SCENARIOS:
        for pol in (POLICIES[0], POLICIES[3]):
            for rep in range(20):
                day = CollectionDay(G, nodes, routes, scen, pol,
                                    np.random.default_rng([1, rep]), np.random.default_rng([2, rep]))
                r = day.run()
                assert r.on_time_l + r.late_l + r.stranded_l == pytest.approx(r.total_l, rel=1e-9)
                assert min(r.on_time_l, r.late_l, r.stranded_l) >= -1e-6
