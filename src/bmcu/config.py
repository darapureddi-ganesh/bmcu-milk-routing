"""Model parameters.

Every number here is an illustrative assumption for a rural dairy collection
network around Tirupati (Chittoor region, Andhra Pradesh). None of it is
company data. Change a value here and the whole pipeline re-runs on it.

Time is measured in minutes from 05:00 (t = 0 is 05:00).
"""
from dataclasses import dataclass, field

# Dairy plant (illustrative location near Tirupati)
PLANT_NAME = "Dairy plant (Tirupati)"
PLANT_LAT, PLANT_LON = 13.6288, 79.4192


@dataclass(frozen=True)
class ScenarioConfig:
    seed: int = 42
    n_bmcus: int = 30                 # bulk milk cooling units to collect from
    n_junctions: int = 14             # road junction / village nodes (no milk)
    radius_km: float = 50.0           # collection area radius around the plant
    n_clusters: int = 5               # BMCUs cluster around mandal centres
    volume_mean_l: float = 2400.0     # mean daily chilled volume per BMCU (litres)
    volume_min_l: float = 800.0
    volume_max_l: float = 4500.0
    ready_min: int = 60               # 06:00 morning milk chilled and ready
    window_min: int = 240             # shortest quality window after ready (minutes)
    window_max: int = 330             # longest quality window after ready (minutes)


@dataclass(frozen=True)
class FleetConfig:
    n_tankers: int = 12
    capacity_l: int = 10_000          # insulated milk tanker capacity
    speed_kmph: float = 38.0          # average rural road speed
    circuity: float = 1.30            # road distance / straight-line distance
    service_fixed_min: float = 10.0   # docking, sampling, paperwork per stop
    pump_l_per_min: float = 250.0     # pumping rate at a BMCU
    plant_close_min: int = 450        # tankers back at plant by 12:30
    fixed_cost_inr: float = 3000.0    # cost of putting one tanker on the road for a day
    cost_per_km_inr: float = 32.0     # fuel + wear + driver per km


@dataclass(frozen=True)
class EconomicsConfig:
    milk_value_inr_per_l: float = 40.0   # value of raw milk to the dairy
    late_value_loss_share: float = 0.30  # share of value lost when collected after its quality window
    overtime_inr_per_min: float = 8.0    # driver/crew overtime after plant close
    standby_fixed_inr: float = 2000.0    # retainer for a standby tanker kept on call


@dataclass(frozen=True)
class DisruptionScenario:
    name: str
    travel_factor: float        # multiplies all travel times (rain slows traffic)
    travel_noise_sd: float      # lognormal sd of trip-level travel time noise
    volume_noise_sd: float      # lognormal sd of daily BMCU volume vs plan
    crossing_close_prob: float  # chance each river crossing (causeway) is closed
    breakdown_prob: float       # chance a tanker breaks down during its route


SCENARIOS = (
    DisruptionScenario("Normal day", 1.00, 0.12, 0.10, 0.00, 0.02),
    DisruptionScenario("Heavy rain", 1.30, 0.22, 0.12, 0.35, 0.04),
    DisruptionScenario("Severe flood", 1.50, 0.30, 0.12, 0.80, 0.06),
)


@dataclass(frozen=True)
class Policy:
    name: str
    capacity_share: float     # share of tanker capacity the plan may use
    planning_time_factor: float  # travel-time inflation used when planning
    standby_tanker: bool      # keep one tanker on call to recover missed milk


POLICIES = (
    Policy("P0 Cost-optimal", 1.00, 1.00, False),
    Policy("P1 Capacity buffer", 0.85, 1.00, False),
    Policy("P2 Buffer + time slack", 0.85, 1.25, False),
    Policy("P3 Buffer + standby tanker", 0.85, 1.00, True),
)

N_REPLICATIONS = 500
SIM_SEED = 2026
