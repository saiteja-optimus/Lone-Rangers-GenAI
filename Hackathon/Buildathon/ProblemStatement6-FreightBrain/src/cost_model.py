"""
FreightBrain cost model — 8-component per-load P&L engine.

All dollar figures in USD. All distances in miles.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIESEL_PRICE_PER_GALLON: float = 3.85
BASE_MPG: float = 6.5
DRIVER_CPM: float = 0.55
INSURANCE_CPM: float = 0.08
MAINTENANCE_CPM: float = 0.15
TOLL_CPM: float = 0.04
FUEL_CPM: float = DIESEL_PRICE_PER_GALLON / BASE_MPG
DEADHEAD_FUEL_CPM: float = DIESEL_PRICE_PER_GALLON / (BASE_MPG * 1.15)

EQUIPMENT_SURCHARGE: dict[str, float] = {
    "Dry Van": 0.00,
    "Reefer": 0.18,
    "Flatbed": 0.10,
    "Step Deck": 0.12,
    "Conestoga": 0.14,
    "RGN": 0.20,
    "Lowboy": 0.20,
    "Power Only": -0.05,
}

MAX_REPO_PENALTY: float = 200.0


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------
def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two WGS-84 coordinate pairs."""
    if any(math.isnan(v) for v in [lat1, lon1, lat2, lon2]):
        return 0.0
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------
@dataclass
class CostBreakdown:
    fuel_cost: float = 0.0
    driver_pay: float = 0.0
    insurance: float = 0.0
    maintenance: float = 0.0
    tolls: float = 0.0
    deadhead_cost: float = 0.0
    repo_penalty: float = 0.0
    equipment_surcharge: float = 0.0

    @property
    def total_cost(self) -> float:
        return (self.fuel_cost + self.driver_pay + self.insurance +
                self.maintenance + self.tolls + self.deadhead_cost +
                self.repo_penalty + self.equipment_surcharge)


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------
def calculate_net_profit(
    gross_rate: float,
    loaded_miles: float,
    deadhead_miles: float = 0.0,
    equipment: str = "Dry Van",
    dest_mls: float = 50.0,
) -> tuple[float, CostBreakdown]:
    """Return (net_profit, CostBreakdown) for a load."""
    surcharge_per_mile = EQUIPMENT_SURCHARGE.get(equipment, 0.0)

    fuel = loaded_miles * FUEL_CPM
    driver = loaded_miles * DRIVER_CPM
    insurance = loaded_miles * INSURANCE_CPM
    maintenance = loaded_miles * MAINTENANCE_CPM
    tolls = loaded_miles * TOLL_CPM
    equip_surcharge = loaded_miles * surcharge_per_mile
    dh_cost = deadhead_miles * (DEADHEAD_FUEL_CPM + DRIVER_CPM * 0.5)
    repo = MAX_REPO_PENALTY * (1.0 - dest_mls / 100.0)

    breakdown = CostBreakdown(
        fuel_cost=round(fuel, 2),
        driver_pay=round(driver, 2),
        insurance=round(insurance, 2),
        maintenance=round(maintenance, 2),
        tolls=round(tolls, 2),
        deadhead_cost=round(dh_cost, 2),
        repo_penalty=round(repo, 2),
        equipment_surcharge=round(equip_surcharge, 2),
    )
    net = gross_rate - breakdown.total_cost
    return round(net, 2), breakdown


def net_rpm(
    gross_rate: float,
    loaded_miles: float,
    deadhead_miles: float = 0.0,
    equipment: str = "Dry Van",
    dest_mls: float = 50.0,
) -> float:
    """Net revenue per loaded mile."""
    if loaded_miles <= 0:
        return 0.0
    n, _ = calculate_net_profit(gross_rate, loaded_miles, deadhead_miles, equipment, dest_mls)
    return round(n / loaded_miles, 4)


def fuel_cost_for_miles(miles: float, equipment: str = "Dry Van") -> float:
    """Standalone loaded fuel cost."""
    return round(miles * FUEL_CPM, 2)
