"""FreightBrain AI Agent — NVIDIA NIM-powered load recommendation."""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
_NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")


@dataclass
class LoadRecommendation:
    load_id: str
    origin: str
    destination: str
    miles: float
    gross_rate: float
    net_profit: float
    net_rpm: float
    dest_mls: float
    reasoning: str
    risk_flags: list
    cost_breakdown: dict


class FreightBrainAgent:
    MAX_TOKENS = 1500

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self._model = _NVIDIA_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # noqa: PLC0415
                if self._api_key:
                    self._client = OpenAI(
                        base_url=_NVIDIA_BASE_URL,
                        api_key=self._api_key,
                    )
            except ImportError:
                logger.warning("openai package not installed — run: pip install openai")
        return self._client

    def recommend(
        self,
        driver_city: str,
        driver_state: str,
        equipment: str,
        candidate_loads: pd.DataFrame,
        mls_df: Optional[pd.DataFrame] = None,
    ) -> tuple[Optional[LoadRecommendation], str]:
        """Return (LoadRecommendation, full_text). Falls back to mock if API unavailable."""
        if candidate_loads.empty:
            return None, "No candidate loads found within the specified deadhead radius."

        loads_str = self._format_loads(candidate_loads.head(5))
        prompt = self._build_prompt(driver_city, driver_state, equipment, loads_str)
        client = self._get_client()
        if client is None or not self._api_key:
            return self._mock_rec(candidate_loads), self._mock_text(candidate_loads)

        try:
            response = client.chat.completions.create(
                model=self._model,
                max_tokens=self.MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
            return self._parse_response(text, candidate_loads), text
        except Exception as exc:
            logger.error("NVIDIA API error: %s", exc)
            return self._mock_rec(candidate_loads), self._mock_text(candidate_loads)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _format_loads(self, df: pd.DataFrame) -> str:
        rows = []
        for _, r in df.iterrows():
            rpm = r["gross_rate"] / r["miles"] if r["miles"] > 0 else 0
            rows.append(
                f"  {r['load_id']}: {r['origin_city']},{r['origin_state']} -> "
                f"{r['dest_city']},{r['dest_state']} | {r['miles']:.0f}mi | "
                f"${r['gross_rate']:.0f} gross | ${r.get('net_profit', 0):.0f} net | "
                f"${rpm:.2f} RPM | DH:{r.get('deadhead_miles', 0):.0f}mi | "
                f"MLS:{r.get('dest_mls', 50):.0f}"
            )
        return "\n".join(rows)

    def _build_prompt(self, city: str, state: str, equipment: str, loads_str: str) -> str:
        return (
            f"You are FreightBrain, an AI load advisor for a small trucking carrier.\n\n"
            f"Driver: {city}, {state} | Equipment: {equipment}\n\n"
            f"Candidate loads (ranked by net profit):\n{loads_str}\n\n"
            "Respond with:\n"
            "**RECOMMENDED LOAD:** [Load ID]\n"
            "**WHY THIS LOAD:** [2-3 sentences on net profit, RPM, destination quality]\n"
            "**RISK FLAGS:** [bullet list of concerns]\n"
            "**RUNNER-UP:** [Load ID and one sentence]\n"
            "**MARKET INSIGHT:** [One sentence on destination market outlook]\n\n"
            "Be direct. Focus on net profit after all costs, RPM, and next-load positioning."
        )

    def _parse_response(self, text: str, candidates: pd.DataFrame) -> LoadRecommendation:
        from cost_model import calculate_net_profit, net_rpm as calc_rpm  # noqa: PLC0415

        best = candidates.iloc[0]
        net, bd = calculate_net_profit(
            float(best["gross_rate"]),
            float(best["miles"]),
            float(best.get("deadhead_miles", 0)),
            str(best.get("equipment", "Dry Van")),
            float(best.get("dest_mls", 50)),
        )
        return LoadRecommendation(
            load_id=str(best["load_id"]),
            origin=f"{best['origin_city']}, {best['origin_state']}",
            destination=f"{best['dest_city']}, {best['dest_state']}",
            miles=float(best["miles"]),
            gross_rate=float(best["gross_rate"]),
            net_profit=net,
            net_rpm=calc_rpm(
                float(best["gross_rate"]),
                float(best["miles"]),
                float(best.get("deadhead_miles", 0)),
                str(best.get("equipment", "Dry Van")),
                float(best.get("dest_mls", 50)),
            ),
            dest_mls=float(best.get("dest_mls", 50)),
            reasoning=text,
            risk_flags=[],
            cost_breakdown=asdict(bd),
        )

    def _mock_rec(self, candidates: pd.DataFrame) -> LoadRecommendation:
        from cost_model import calculate_net_profit, net_rpm as calc_rpm  # noqa: PLC0415

        best = candidates.iloc[0]
        net, bd = calculate_net_profit(
            float(best["gross_rate"]),
            float(best["miles"]),
            float(best.get("deadhead_miles", 0)),
            str(best.get("equipment", "Dry Van")),
            float(best.get("dest_mls", 50)),
        )
        return LoadRecommendation(
            load_id=str(best["load_id"]),
            origin=f"{best['origin_city']}, {best['origin_state']}",
            destination=f"{best['dest_city']}, {best['dest_state']}",
            miles=float(best["miles"]),
            gross_rate=float(best["gross_rate"]),
            net_profit=net,
            net_rpm=calc_rpm(
                float(best["gross_rate"]),
                float(best["miles"]),
                float(best.get("deadhead_miles", 0)),
                str(best.get("equipment", "Dry Van")),
                float(best.get("dest_mls", 50)),
            ),
            dest_mls=float(best.get("dest_mls", 50)),
            reasoning=self._mock_text(candidates),
            risk_flags=["Add NVIDIA_API_KEY to .env for live Claude AI analysis"],
            cost_breakdown=asdict(bd),
        )

    def _mock_text(self, candidates: pd.DataFrame) -> str:
        best = candidates.iloc[0]
        rpm = best["gross_rate"] / best["miles"] if best["miles"] > 0 else 0
        runner = candidates.iloc[1]["load_id"] if len(candidates) > 1 else "N/A"
        return (
            f"**RECOMMENDED LOAD:** {best['load_id']}\n\n"
            f"**WHY THIS LOAD:** This {best['miles']:.0f}-mile run from "
            f"{best['origin_city']}, {best['origin_state']} to "
            f"{best['dest_city']}, {best['dest_state']} generates "
            f"${best['gross_rate']:.0f} gross at ${rpm:.2f}/mi — the strongest "
            "net-profit opportunity in your deadhead radius. The destination market "
            "liquidity score indicates healthy outbound load availability, reducing "
            "your repositioning risk.\n\n"
            "**RISK FLAGS:**\n"
            "- Add NVIDIA_API_KEY to .env to enable full AI analysis\n"
            "- Confirm rate with broker before dispatching\n\n"
            f"**RUNNER-UP:** {runner} — second-best net profit in radius.\n\n"
            "**MARKET INSIGHT:** Demand along this corridor is steady near the "
            "30-day average, making this a reliable rather than exceptional opportunity.\n\n"
            "*FreightBrain Demo Mode — live Claude reasoning requires NVIDIA_API_KEY*"
        )
