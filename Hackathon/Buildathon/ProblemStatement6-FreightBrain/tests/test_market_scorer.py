"""Unit tests for FreightBrain Market Liquidity Scorer."""
import sys
import os
import itertools
import random
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from market_scorer import (
    compute_market_liquidity_scores,
    get_top_markets,
    get_mls_for_city,
    compute_lane_balance,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Synthetic loads: 3 hub cities (many loads) + 2 rural (few loads)."""
    hubs = [
        ("Chicago", "IL", 41.8781, -87.6298),
        ("Dallas", "TX", 32.7767, -96.7970),
        ("Atlanta", "GA", 33.7490, -84.3880),
    ]
    rural = [
        ("Podunk", "WY", 43.0, -107.0),
        ("Smalltown", "MT", 46.0, -111.0),
    ]
    rng = random.Random(0)
    rows = []
    lid = 0
    for (oc, os_, olat, olon), (dc, ds, dlat, dlon) in itertools.product(hubs, hubs):
        if oc == dc:
            continue
        for _ in range(20):
            mi = rng.randint(500, 1200)
            rows.append({
                "load_id": f"L{lid:05d}", "origin_city": oc, "origin_state": os_,
                "dest_city": dc, "dest_state": ds, "miles": mi,
                "gross_rate": mi * rng.uniform(1.8, 2.5), "equipment": "Dry Van",
                "origin_lat": olat, "origin_lon": olon,
                "dest_lat": dlat, "dest_lon": dlon,
            })
            lid += 1
    for oc, os_, olat, olon in rural:
        dc, ds, dlat, dlon = hubs[0]
        for _ in range(2):
            rows.append({
                "load_id": f"L{lid:05d}", "origin_city": oc, "origin_state": os_,
                "dest_city": dc, "dest_state": ds, "miles": 800,
                "gross_rate": 1400.0, "equipment": "Dry Van",
                "origin_lat": olat, "origin_lon": olon,
                "dest_lat": dlat, "dest_lon": dlon,
            })
            lid += 1
    return pd.DataFrame(rows)


class TestMLSBounds:
    def test_scores_between_0_and_100(self, sample_df):
        mls = compute_market_liquidity_scores(sample_df)
        assert (mls["mls_score"] >= 0).all()
        assert (mls["mls_score"] <= 100).all()

    def test_empty_dataframe_returns_empty(self):
        assert compute_market_liquidity_scores(pd.DataFrame()).empty

    def test_all_cities_present(self, sample_df):
        mls = compute_market_liquidity_scores(sample_df)
        cities = set(mls["city"])
        assert "Chicago" in cities
        assert "Dallas" in cities
        assert "Atlanta" in cities


class TestLaneBalance:
    def test_perfect_balance(self):
        assert compute_lane_balance(100, 100) == 1.0

    def test_total_imbalance(self):
        assert compute_lane_balance(100, 0) == 0.0

    def test_partial_imbalance(self):
        score = compute_lane_balance(80, 40)
        assert 0 < score < 1

    def test_zero_zero(self):
        assert compute_lane_balance(0, 0) == 1.0


class TestHubVsRural:
    def test_hub_cities_beat_rural(self, sample_df):
        mls = compute_market_liquidity_scores(sample_df)
        hub_avg = mls[mls["city"].isin(["Chicago", "Dallas", "Atlanta"])]["mls_score"].mean()
        rural_avg = mls[mls["city"].isin(["Podunk", "Smalltown"])]["mls_score"].mean()
        assert hub_avg > rural_avg, f"Hub {hub_avg:.1f} not > rural {rural_avg:.1f}"

    def test_top_market_is_a_hub(self, sample_df):
        mls = compute_market_liquidity_scores(sample_df)
        assert mls.iloc[0]["city"] in ["Chicago", "Dallas", "Atlanta"]


class TestGetTopMarkets:
    def test_returns_exactly_n(self, sample_df):
        mls = compute_market_liquidity_scores(sample_df)
        assert len(get_top_markets(mls, 5)) == 5

    def test_returns_all_when_n_exceeds_length(self, sample_df):
        mls = compute_market_liquidity_scores(sample_df)
        assert len(get_top_markets(mls, 100)) == len(mls)

    def test_sorted_descending(self, sample_df):
        mls = compute_market_liquidity_scores(sample_df)
        scores = get_top_markets(mls, len(mls))["mls_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_default_n_10(self, sample_df):
        mls = compute_market_liquidity_scores(sample_df)
        assert len(get_top_markets(mls)) == min(10, len(mls))


class TestGetMLSForCity:
    def test_known_city(self, sample_df):
        mls = compute_market_liquidity_scores(sample_df)
        score = get_mls_for_city(mls, "Chicago", "IL")
        assert 0 <= score <= 100

    def test_unknown_city_returns_50(self, sample_df):
        mls = compute_market_liquidity_scores(sample_df)
        assert get_mls_for_city(mls, "Nowhere", "ZZ") == 50.0

    def test_case_insensitive(self, sample_df):
        mls = compute_market_liquidity_scores(sample_df)
        assert get_mls_for_city(mls, "chicago", "il") == get_mls_for_city(mls, "Chicago", "IL")
