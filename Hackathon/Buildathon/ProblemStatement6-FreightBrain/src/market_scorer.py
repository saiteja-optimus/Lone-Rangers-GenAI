"""FreightBrain Market Liquidity Score (MLS 0-100)."""
from __future__ import annotations
import pandas as pd
import numpy as np


def compute_market_liquidity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-city MLS composite score from a load DataFrame."""
    if df.empty:
        return pd.DataFrame()

    outbound = (
        df.groupby(["origin_city", "origin_state"])
        .agg(
            outbound_loads=("load_id", "count"),
            avg_gross_rate=("gross_rate", "mean"),
            avg_miles=("miles", "mean"),
            lat=("origin_lat", "first"),
            lon=("origin_lon", "first"),
        )
        .reset_index()
        .rename(columns={"origin_city": "city", "origin_state": "state"})
    )

    inbound = (
        df.groupby(["dest_city", "dest_state"])
        .size()
        .reset_index(name="inbound_loads")
        .rename(columns={"dest_city": "city", "dest_state": "state"})
    )

    diversity = (
        df.groupby(["origin_city", "origin_state"])["dest_state"]
        .nunique()
        .reset_index(name="unique_dest_states")
        .rename(columns={"origin_city": "city", "origin_state": "state"})
    )

    merged = outbound.merge(inbound, on=["city", "state"], how="left")
    merged = merged.merge(diversity, on=["city", "state"], how="left")
    merged["inbound_loads"] = merged["inbound_loads"].fillna(0)
    merged["unique_dest_states"] = merged["unique_dest_states"].fillna(1)
    merged["avg_rpm"] = merged.apply(
        lambda r: r["avg_gross_rate"] / r["avg_miles"] if r["avg_miles"] > 0 else 0.0,
        axis=1,
    )
    merged["lane_balance"] = merged.apply(
        lambda r: 1.0 - abs(r["inbound_loads"] - r["outbound_loads"])
        / max(r["inbound_loads"], r["outbound_loads"], 1),
        axis=1,
    )

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else pd.Series([0.5] * len(s), index=s.index)

    merged["n_outbound"] = _norm(merged["outbound_loads"])
    merged["n_balance"] = merged["lane_balance"]
    merged["n_rpm"] = _norm(merged["avg_rpm"])
    merged["n_diversity"] = _norm(merged["unique_dest_states"])

    merged["mls_score"] = (
        0.35 * merged["n_outbound"]
        + 0.25 * merged["n_balance"]
        + 0.25 * merged["n_rpm"]
        + 0.15 * merged["n_diversity"]
    ) * 100
    merged["mls_score"] = merged["mls_score"].clip(0, 100).round(1)
    merged["grade"] = merged["mls_score"].apply(_grade)
    merged = merged.drop(
        columns=["n_outbound", "n_balance", "n_rpm", "n_diversity", "avg_gross_rate", "avg_miles"],
        errors="ignore",
    )
    return merged.sort_values("mls_score", ascending=False).reset_index(drop=True)


def get_top_markets(mls_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-n markets by MLS score."""
    return mls_df.head(n).reset_index(drop=True)


def get_mls_for_city(mls_df: pd.DataFrame, city: str, state: str) -> float:
    """Return MLS score for city/state, default 50.0 if unknown."""
    mask = (mls_df["city"].str.lower() == city.strip().lower()) & (
        mls_df["state"].str.upper() == state.strip().upper()
    )
    matches = mls_df[mask]
    return float(matches.iloc[0]["mls_score"]) if not matches.empty else 50.0


def compute_lane_balance(outbound: int, inbound: int) -> float:
    """Return lane balance ratio (0=fully imbalanced, 1=perfectly balanced)."""
    if max(outbound, inbound) == 0:
        return 1.0
    return 1.0 - abs(outbound - inbound) / max(outbound, inbound)


def _grade(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    if score >= 20:
        return "D"
    return "F"
