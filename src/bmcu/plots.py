"""Static figures for the README (matplotlib)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402

from .scenario import RIVER  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
})


def route_map(G, nodes: pd.DataFrame, routes: list[list[int]], path: str, title: str):
    pos = nx_pos(G)
    fig, ax = plt.subplots(figsize=(8.5, 8))
    for u, v, d in G.edges(data=True):
        (x1, y1), (x2, y2) = pos[u], pos[v]
        if d["bridge"]:
            ax.plot([x1, x2], [y1, y2], color=INK, lw=3, zorder=2, solid_capstyle="round")
        elif d["crossing"]:
            ax.plot([x1, x2], [y1, y2], color=MUTED, lw=1, ls=(0, (3, 2)), zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], color=GRID, lw=1, zorder=1)
    (rx1, ry1), (rx2, ry2) = RIVER
    ax.plot([rx1, rx2], [ry1, ry2], color="#7fb2e8", lw=6, alpha=0.5, zorder=0, solid_capstyle="round")
    ax.text(rx2 - 2, ry2 + 1.5, "River (illustrative)", color=INK_2, ha="right", fontsize=9)

    for k, r in enumerate(routes, 1):
        stops = [0] + r + [0]
        seq = [stops[0]]
        for a, b in zip(stops, stops[1:]):  # follow the actual roads between stops
            seq += nx.shortest_path(G, a, b, weight="length_km")[1:]
        xs = [pos[n][0] for n in seq]
        ys = [pos[n][1] for n in seq]
        ax.plot(xs, ys, color=BLUE, lw=2, alpha=0.85, zorder=3)
        mx, my = pos[r[len(r) // 2]]
        ax.text(mx + 1.2, my + 1.2, f"R{k}", fontsize=8.5, color=INK, weight="bold", zorder=6)

    j = nodes[nodes.kind == "junction"]
    ax.scatter(j.x_km, j.y_km, s=14, color=MUTED, zorder=4, label="Road junction")
    b = nodes[nodes.kind == "bmcu"]
    ax.scatter(b.x_km, b.y_km, s=b.volume_l / 25, color=AQUA, edgecolor=SURFACE, lw=1.5, zorder=5,
               label="BMCU (size = daily litres)")
    ax.scatter([0], [0], s=180, marker="s", color=INK, zorder=6, label="Dairy plant")
    ax.plot([], [], color=BLUE, lw=2, label="Planned tanker route")
    ax.plot([], [], color=MUTED, lw=1, ls=(0, (3, 2)), label="Causeway (can flood)")
    ax.plot([], [], color=INK, lw=3, label="Highway bridge (stays open)")
    ax.set_aspect("equal")
    ax.set_xlabel("km east of plant")
    ax.set_ylabel("km north of plant")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def nx_pos(G):
    return {n: d["pos"] for n, d in G.nodes(data=True)}


def criticality_chart(crit: pd.DataFrame, path: str, top: int = 10):
    d = crit.head(top).iloc[::-1]
    labels = [f"{n}{'  [highway bridge]' if b else ('  [causeway]' if c else '')}"
              for n, c, b in zip(d.name, d.crossing, d.bridge)]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(labels, d.extra_km, color=BLUE, height=0.6)
    for y, v in enumerate(d.extra_km):
        ax.text(v + 3, y, f"+{v:.0f} km", va="center", fontsize=8.5, color=INK_2)
    ax.set_xlabel("Extra plant-to-BMCU km if this link closes (sum over all BMCUs)")
    ax.set_title("Most critical road links: losing one adds the most distance", loc="left")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cost_breakdown_chart(summary: pd.DataFrame, path: str):
    scenarios = list(dict.fromkeys(summary.scenario))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(12, 3.9), sharex=True, sharey=True)
    xmax = summary.total_cost_inr.max() / 1000 * 1.18
    for ax, scen in zip(axes, scenarios):
        d = summary[summary.scenario == scen].iloc[::-1]
        y = range(len(d))
        t = d.transport_inr / 1000
        m = (d.total_cost_inr - d.transport_inr) / 1000
        ax.barh(y, t, color=BLUE, height=0.6, label="Transport")
        ax.barh(y, m, left=t + 0.6, color=ORANGE, height=0.6, label="Milk lost or degraded")
        for yi, tot in zip(y, d.total_cost_inr / 1000):
            ax.text(tot + 3, yi, f"₹{tot:.0f}k", va="center", fontsize=8.5, color=INK_2)
        ax.set_yticks(list(y), [p.split(" ", 1)[1] for p in d.policy], fontsize=9)
        ax.set_title(scen, loc="left")
        ax.set_xlim(0, xmax)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Expected cost per day (₹ thousand)")
    axes[0].legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.suptitle("Cheapest plan on paper is the most expensive once disruption hits",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def risk_chart(summary: pd.DataFrame, path: str):
    scenarios = list(dict.fromkeys(summary.scenario))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(12, 3.6), sharex=True, sharey=True)
    for ax, scen in zip(axes, scenarios):
        d = summary[summary.scenario == scen].iloc[::-1]
        y = list(range(len(d)))
        ax.errorbar(d.at_risk_l, y, xerr=d.at_risk_ci, fmt="o", color=BLUE, ms=8,
                    ecolor=MUTED, elinewidth=1.5, capsize=3)
        for yi, v in zip(y, d.at_risk_l):
            ax.text(v, yi + 0.28, f"{v:,.0f} L", ha="center", fontsize=8.5, color=INK_2)
        ax.set_yticks(y, [p.split(" ", 1)[1] for p in d.policy], fontsize=9)
        ax.set_title(scen, loc="left")
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Milk at risk per day (litres, mean ± 95% CI)")
        ax.set_ylim(-0.6, len(d) - 0.3)
    fig.suptitle("Litres collected late or stranded, by planning policy",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
