"""Generate an illustrative BMCU collection scenario.

Nodes are placed on a local flat grid in kilometres around the plant
(x = east, y = north) and converted to latitude/longitude for maps.
"""
import math

import numpy as np
import pandas as pd

from .config import PLANT_LAT, PLANT_LON, PLANT_NAME, ScenarioConfig

KM_PER_DEG_LAT = 110.574


def km_to_latlon(x_km: float, y_km: float) -> tuple[float, float]:
    lat = PLANT_LAT + y_km / KM_PER_DEG_LAT
    lon = PLANT_LON + x_km / (111.320 * math.cos(math.radians(PLANT_LAT)))
    return lat, lon


# An illustrative river running west-south-west to east-north-east,
# about 8 km north of the plant. Road links that cross it are causeways
# or low bridges that can close in heavy rain.
RIVER = ((-60.0, 2.0), (60.0, 16.0))


def river_side(x: float, y: float) -> int:
    (x1, y1), (x2, y2) = RIVER
    cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    return 1 if cross > 0 else -1


def _random_in_disc(rng, radius, n):
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return r * np.cos(a), r * np.sin(a)


def generate(cfg: ScenarioConfig = ScenarioConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows = [dict(node=0, name=PLANT_NAME, kind="plant", x_km=0.0, y_km=0.0,
                 volume_l=0.0, ready_min=0, deadline_min=10_000)]

    # BMCUs cluster around a few mandal centres
    ang = rng.uniform(0, 2 * np.pi, cfg.n_clusters)
    dist = rng.uniform(15, cfg.radius_km - 8, cfg.n_clusters)
    centres = np.c_[dist * np.cos(ang), dist * np.sin(ang)]
    cluster = rng.integers(0, cfg.n_clusters, cfg.n_bmcus)

    sigma = 0.35
    mu = math.log(cfg.volume_mean_l) - sigma ** 2 / 2
    for i in range(cfg.n_bmcus):
        cx, cy = centres[cluster[i]]
        x, y = cx + rng.normal(0, 6), cy + rng.normal(0, 6)
        # keep inside the collection area
        d = math.hypot(x, y)
        if d > cfg.radius_km:
            x, y = x * cfg.radius_km / d, y * cfg.radius_km / d
        vol = float(np.clip(rng.lognormal(mu, sigma), cfg.volume_min_l, cfg.volume_max_l))
        ready = cfg.ready_min + int(rng.integers(0, 31))
        window = int(rng.integers(cfg.window_min, cfg.window_max + 1))
        rows.append(dict(node=i + 1, name=f"BMCU-{i + 1:02d}", kind="bmcu",
                         x_km=x, y_km=y, volume_l=round(vol, -1),
                         ready_min=ready, deadline_min=ready + window))

    jx, jy = _random_in_disc(rng, cfg.radius_km, cfg.n_junctions)
    for j in range(cfg.n_junctions):
        rows.append(dict(node=cfg.n_bmcus + 1 + j, name=f"Junction-{j + 1:02d}",
                         kind="junction", x_km=float(jx[j]), y_km=float(jy[j]),
                         volume_l=0.0, ready_min=0, deadline_min=10_000))

    df = pd.DataFrame(rows)
    df[["lat", "lon"]] = [km_to_latlon(x, y) for x, y in zip(df.x_km, df.y_km)]
    df["river_side"] = [river_side(x, y) for x, y in zip(df.x_km, df.y_km)]
    return df
