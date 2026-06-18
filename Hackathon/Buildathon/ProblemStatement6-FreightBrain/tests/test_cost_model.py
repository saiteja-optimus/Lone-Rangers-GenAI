"""Unit tests for FreightBrain cost model."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cost_model import (
    calculate_net_profit,
    fuel_cost_for_miles,
    haversine_miles,
    FUEL_CPM,
    DRIVER_CPM,
    MAX_REPO_PENALTY,
)


class TestFuelCost:
    def test_500_miles_approx_296(self):
        """500 miles at diesel $3.85 / 6.5 mpg = $0.5923/mi -> ~$296."""
        cost = fuel_cost_for_miles(500)
        assert 285 < cost < 310, f"Expected ~$296, got ${cost}"

    def test_zero_miles(self):
        assert fuel_cost_for_miles(0) == 0.0

    def test_proportional(self):
        assert abs(fuel_cost_for_miles(1000) - 2 * fuel_cost_for_miles(500)) < 0.02


class TestAtlantaFortWorth:
    """908 miles, $1700 gross, 0 deadhead, MLS 50."""
    MILES, GROSS = 908.0, 1700.0

    def test_net_in_range(self):
        net, _ = calculate_net_profit(self.GROSS, self.MILES, dest_mls=50)
        assert 250 <= net <= 400, f"Expected $280-$320 range, got ${net}"

    def test_components_positive(self):
        _, bd = calculate_net_profit(self.GROSS, self.MILES, dest_mls=50)
        assert bd.fuel_cost > 0
        assert bd.driver_pay > 0
        assert bd.insurance > 0
        assert bd.maintenance > 0
        assert bd.tolls > 0

    def test_gross_minus_cost_equals_net(self):
        net, bd = calculate_net_profit(self.GROSS, self.MILES, dest_mls=50)
        assert abs((self.GROSS - bd.total_cost) - net) < 0.02

    def test_net_less_than_gross(self):
        net, _ = calculate_net_profit(self.GROSS, self.MILES, dest_mls=50)
        assert net < self.GROSS


class TestRepoPenalty:
    def test_mls_0_max_penalty(self):
        _, bd = calculate_net_profit(1700, 908, dest_mls=0)
        assert abs(bd.repo_penalty - MAX_REPO_PENALTY) < 0.01

    def test_mls_100_zero_penalty(self):
        _, bd = calculate_net_profit(1700, 908, dest_mls=100)
        assert bd.repo_penalty == 0.0

    def test_mls_50_half_penalty(self):
        _, bd = calculate_net_profit(1700, 908, dest_mls=50)
        assert abs(bd.repo_penalty - 100.0) < 0.01

    def test_monotonically_decreasing(self):
        penalties = [
            calculate_net_profit(1700, 908, dest_mls=m)[1].repo_penalty
            for m in [0, 25, 50, 75, 100]
        ]
        assert all(penalties[i] >= penalties[i + 1] for i in range(len(penalties) - 1))


class TestDeadheadImpact:
    def test_deadhead_reduces_net(self):
        net0, _ = calculate_net_profit(1700, 908, deadhead_miles=0, dest_mls=50)
        net100, _ = calculate_net_profit(1700, 908, deadhead_miles=100, dest_mls=50)
        assert net100 < net0

    def test_deadhead_cost_positive(self):
        _, bd = calculate_net_profit(1700, 908, deadhead_miles=100, dest_mls=50)
        assert bd.deadhead_cost > 0

    def test_zero_deadhead_zero_cost(self):
        _, bd = calculate_net_profit(1700, 908, deadhead_miles=0, dest_mls=50)
        assert bd.deadhead_cost == 0.0

    def test_deadhead_scales_with_distance(self):
        _, bd100 = calculate_net_profit(1700, 908, deadhead_miles=100, dest_mls=50)
        _, bd200 = calculate_net_profit(1700, 908, deadhead_miles=200, dest_mls=50)
        assert bd200.deadhead_cost > bd100.deadhead_cost


class TestEquipmentSurcharges:
    MILES, GROSS = 500.0, 1500.0

    def test_reefer_more_expensive_than_dry_van(self):
        _, bd_dv = calculate_net_profit(self.GROSS, self.MILES, equipment="Dry Van", dest_mls=50)
        _, bd_rf = calculate_net_profit(self.GROSS, self.MILES, equipment="Reefer", dest_mls=50)
        assert bd_rf.total_cost > bd_dv.total_cost

    def test_flatbed_more_expensive_than_dry_van(self):
        _, bd_dv = calculate_net_profit(self.GROSS, self.MILES, equipment="Dry Van", dest_mls=50)
        _, bd_fb = calculate_net_profit(self.GROSS, self.MILES, equipment="Flatbed", dest_mls=50)
        assert bd_fb.total_cost > bd_dv.total_cost

    def test_dry_van_zero_surcharge(self):
        _, bd = calculate_net_profit(self.GROSS, self.MILES, equipment="Dry Van", dest_mls=50)
        assert bd.equipment_surcharge == 0.0

    def test_reefer_surcharge_90_dollars(self):
        """Reefer 0.18/mi * 500 mi = $90."""
        _, bd = calculate_net_profit(self.GROSS, self.MILES, equipment="Reefer", dest_mls=50)
        assert abs(bd.equipment_surcharge - 90.0) < 0.5


class TestHaversine:
    def test_atlanta_to_fort_worth(self):
        dist = haversine_miles(33.749, -84.388, 32.755, -97.331)
        assert 750 < dist < 830, f"Got {dist:.1f} mi"

    def test_same_point_zero(self):
        assert haversine_miles(40.0, -75.0, 40.0, -75.0) == 0.0

    def test_nyc_to_la(self):
        dist = haversine_miles(40.7128, -74.006, 34.0522, -118.2437)
        assert 2400 < dist < 2500
