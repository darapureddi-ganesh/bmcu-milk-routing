"""Road network model and vulnerability analysis.

The road network is a proxy: each node links to its nearest neighbours,
plus a minimum spanning tree so that everything is connected. Road length
is straight-line distance times a circuity factor. Links that cross the
river are causeways that can close in heavy rain; one of them is a
high-level highway bridge that stays open.
"""
import itertools
import math

import networkx as nx
import numpy as np
import pandas as pd

from .config import FleetConfig
from .scenario import RIVER


def _segments_intersect(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def build_graph(nodes: pd.DataFrame, fleet: FleetConfig = FleetConfig(), k: int = 4) -> nx.Graph:
    pts = {int(r.node): (r.x_km, r.y_km) for r in nodes.itertuples()}
    G = nx.Graph()
    for r in nodes.itertuples():
        G.add_node(int(r.node), name=r.name, kind=r.kind, pos=(r.x_km, r.y_km))

    def add_edge(a, b):
        if a == b or G.has_edge(a, b):
            return
        straight = math.dist(pts[a], pts[b])
        length = straight * fleet.circuity
        crossing = _segments_intersect(pts[a], pts[b], *RIVER)
        G.add_edge(a, b, length_km=length, time_min=length / fleet.speed_kmph * 60,
                   crossing=crossing, bridge=False)

    ids = list(pts)
    for a in ids:
        nearest = sorted((math.dist(pts[a], pts[b]), b) for b in ids if b != a)[:k]
        for _, b in nearest:
            add_edge(a, b)

    complete = nx.Graph()
    for a, b in itertools.combinations(ids, 2):
        complete.add_edge(a, b, w=math.dist(pts[a], pts[b]))
    for a, b in nx.minimum_spanning_edges(complete, weight="w", data=False):
        add_edge(a, b)

    # The crossing closest to the plant is the highway bridge: it never closes.
    crossings = [(u, v) for u, v, d in G.edges(data=True) if d["crossing"]]
    if crossings:
        def mid_dist(e):
            (x1, y1), (x2, y2) = pts[e[0]], pts[e[1]]
            return math.hypot((x1 + x2) / 2, (y1 + y2) / 2)
        u, v = min(crossings, key=mid_dist)
        G.edges[u, v]["bridge"] = True
    return G


def closable_crossings(G: nx.Graph) -> list[tuple[int, int]]:
    return [(u, v) for u, v, d in G.edges(data=True) if d["crossing"] and not d["bridge"]]


def shortest_matrices(G: nx.Graph, nodes: list[int], closed=frozenset()):
    """Shortest-path distance (km) and time (min) between the given nodes.

    Unreachable pairs get np.inf.
    """
    H = G.copy()
    H.remove_edges_from(closed)
    n = len(nodes)
    dist = np.full((n, n), np.inf)
    time = np.full((n, n), np.inf)
    index = {v: i for i, v in enumerate(nodes)}
    for i, s in enumerate(nodes):
        lengths, paths = nx.single_source_dijkstra(H, s, weight="length_km")
        for t, d in lengths.items():
            if t in index:
                j = index[t]
                dist[i, j] = d
                path = paths[t]
                time[i, j] = sum(H.edges[a, b]["time_min"] for a, b in zip(path, path[1:]))
    return dist, time


def link_criticality(G: nx.Graph, plant: int, bmcus: list[int]) -> pd.DataFrame:
    """Rank road links by how much the plant-to-BMCU network depends on them.

    betweenness: share of plant-to-BMCU shortest paths that use the link.
    extra_km: increase in total plant-to-BMCU shortest distance if the link closes.
    cut_off: number of BMCUs that can no longer be reached at all.
    """
    bet = nx.edge_betweenness_centrality_subset(G, [plant], bmcus, normalized=False, weight="length_km")
    base = nx.single_source_dijkstra_path_length(G, plant, weight="length_km")
    base_total = sum(base[b] for b in bmcus)
    rows = []
    for u, v, d in G.edges(data=True):
        H = G.copy()
        H.remove_edge(u, v)
        after = nx.single_source_dijkstra_path_length(H, plant, weight="length_km")
        cut = [b for b in bmcus if b not in after]
        reach_total = sum(after[b] for b in bmcus if b in after)
        reach_base = sum(base[b] for b in bmcus if b in after)
        rows.append(dict(u=u, v=v, name=f"{G.nodes[u]['name']} – {G.nodes[v]['name']}",
                         length_km=d["length_km"], crossing=d["crossing"], bridge=d["bridge"],
                         betweenness=bet.get((u, v), bet.get((v, u), 0.0)),
                         extra_km=reach_total - reach_base, cut_off=len(cut)))
    df = pd.DataFrame(rows)
    df["base_total_km"] = base_total
    return df.sort_values(["cut_off", "extra_km", "betweenness"], ascending=False).reset_index(drop=True)
