"""FreightBrain load data parser — handles 3 rotating unstructured text formats."""
from __future__ import annotations
import os
import re
import glob
import logging
import random
import datetime
from typing import Optional
import pandas as pd
import numpy as np

try:
    from city_coords import lookup_coords
except ImportError:
    def lookup_coords(city, state):
        return None

logger = logging.getLogger(__name__)

# --- block boundary detection -----------------------------------------------

_BLOCK_START_RE = re.compile(
    r'^(LOAD\s+L-\d+|===\s*Load\s+Ref\s+L-\d+|L-\d+\s*\|)',
    re.MULTILINE
)


def _split_blocks(text: str) -> list[str]:
    """Split file text into individual load blocks."""
    starts = [m.start() for m in _BLOCK_START_RE.finditer(text)]
    if not starts:
        return []
    blocks = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        blocks.append(text[start:end].strip())
    return blocks


# --- format-specific parsers ------------------------------------------------

# Format A example:
# LOAD L-0000001
#   Origin......: Atlanta, GA 30303  (33.749, -84.388)
#   Destination.: Fort Worth, TX 76102  (32.7555, -97.3308)
#   Pickup......: 2026-06-15 13:00 to 2026-06-15 16:00
#   Delivery....: 2026-06-17 14:00 to 2026-06-17 18:00
#   Equipment...: Dry Van
#   Commodity...: Packaged food / 37,500 lbs
#   Miles.......: 908
#   Rate........: $1,700  ($1.87/mi)
#   Requirements: None

_A_LOAD_ID   = re.compile(r'^LOAD\s+(L-\d+)', re.MULTILINE)
_A_ORIGIN    = re.compile(r'Origin[\.\s]*:\s*([^,]+),\s*([A-Z]{2})\s+\d+\s*(?:\(([-\d.]+),\s*([-\d.]+)\))?')
_A_DEST      = re.compile(r'Destination[\.\s]*:\s*([^,]+),\s*([A-Z]{2})\s+\d+\s*(?:\(([-\d.]+),\s*([-\d.]+)\))?')
_A_PICKUP    = re.compile(r'Pickup[\.\s]*:\s*([\d\-]+ [\d:]+)\s+to\s+([\d\-]+ [\d:]+)')
_A_DELIVERY  = re.compile(r'Delivery[\.\s]*:\s*([\d\-]+ [\d:]+)\s+to\s+([\d\-]+ [\d:]+)')
_A_EQUIP     = re.compile(r'Equipment[\.\s]*:\s*(.+)')
_A_COMMODITY = re.compile(r'Commodity[\.\s]*:\s*([^/]+?)(?:\s*/\s*([\d,]+)\s*lbs)?$', re.MULTILINE)
_A_MILES     = re.compile(r'Miles[\.\s]*:\s*([\d,]+)')
_A_RATE      = re.compile(r'Rate[\.\s]*:\s*\$([\d,]+)')
_A_RPM       = re.compile(r'\(\$([\d.]+)/mi\)')
_A_REQS      = re.compile(r'Requirements[\.\s]*:\s*(.+)')


def _parse_format_a(block: str) -> Optional[dict]:
    m_id = _A_LOAD_ID.search(block)
    if not m_id:
        return None
    m_ori  = _A_ORIGIN.search(block)
    m_dst  = _A_DEST.search(block)
    m_pu   = _A_PICKUP.search(block)
    m_del  = _A_DELIVERY.search(block)
    m_eq   = _A_EQUIP.search(block)
    m_com  = _A_COMMODITY.search(block)
    m_mi   = _A_MILES.search(block)
    m_rate = _A_RATE.search(block)
    m_rpm  = _A_RPM.search(block)
    m_req  = _A_REQS.search(block)
    return {
        "load_id":        m_id.group(1),
        "origin_city":    m_ori.group(1).strip() if m_ori else None,
        "origin_state":   m_ori.group(2).strip() if m_ori else None,
        "origin_lat":     float(m_ori.group(3)) if m_ori and m_ori.group(3) else None,
        "origin_lon":     float(m_ori.group(4)) if m_ori and m_ori.group(4) else None,
        "dest_city":      m_dst.group(1).strip() if m_dst else None,
        "dest_state":     m_dst.group(2).strip() if m_dst else None,
        "dest_lat":       float(m_dst.group(3)) if m_dst and m_dst.group(3) else None,
        "dest_lon":       float(m_dst.group(4)) if m_dst and m_dst.group(4) else None,
        "pickup_start":   m_pu.group(1) if m_pu else None,
        "pickup_end":     m_pu.group(2) if m_pu else None,
        "delivery_start": m_del.group(1) if m_del else None,
        "delivery_end":   m_del.group(2) if m_del else None,
        "equipment_type": m_eq.group(1).strip() if m_eq else "Dry Van",
        "commodity":      m_com.group(1).strip() if m_com else None,
        "weight_lbs":     int(m_com.group(2).replace(",", "")) if m_com and m_com.group(2) else None,
        "distance_miles": float(m_mi.group(1).replace(",", "")) if m_mi else None,
        "rate_usd":       float(m_rate.group(1).replace(",", "")) if m_rate else None,
        "rate_per_mile":  float(m_rpm.group(1)) if m_rpm else None,
        "requirements":   m_req.group(1).strip() if m_req else None,
    }


# Format B example:
# === Load Ref L-0000002 ===
# PU: Knoxville, TN 37902 | 2026-06-18 10:00 - 2026-06-18 13:00
# DEL: Toledo, OH 43604 | 2026-06-19 07:00 - 2026-06-19 13:00
# Equip Flatbed; 518 mi; 43,100# Lumber
# Pay $1,350 @ $2.61/mile
# Notes: None

_B_LOAD_ID   = re.compile(r'===\s*Load\s+Ref\s+(L-\d+)\s*===')
_B_PU        = re.compile(r'^PU:\s*([^,]+),\s*([A-Z]{2})\s+\d+\s*\|\s*([\d\-]+ [\d:]+)\s*-\s*([\d\-]+ [\d:]+)', re.MULTILINE)
_B_DEL       = re.compile(r'^DEL:\s*([^,]+),\s*([A-Z]{2})\s+\d+\s*\|\s*([\d\-]+ [\d:]+)\s*-\s*([\d\-]+ [\d:]+)', re.MULTILINE)
_B_EQUIP_LINE= re.compile(r'^Equip\s+([^;]+);\s*([\d,]+)\s*mi;\s*([\d,]+)#\s*(.+)$', re.MULTILINE)
_B_PAY       = re.compile(r'Pay\s+\$([\d,]+)\s+@\s+\$([\d.]+)/mile')
_B_NOTES     = re.compile(r'Notes:\s*(.+)')


def _parse_format_b(block: str) -> Optional[dict]:
    m_id  = _B_LOAD_ID.search(block)
    if not m_id:
        return None
    m_pu  = _B_PU.search(block)
    m_del = _B_DEL.search(block)
    m_eq  = _B_EQUIP_LINE.search(block)
    m_pay = _B_PAY.search(block)
    m_nt  = _B_NOTES.search(block)
    return {
        "load_id":        m_id.group(1),
        "origin_city":    m_pu.group(1).strip() if m_pu else None,
        "origin_state":   m_pu.group(2).strip() if m_pu else None,
        "origin_lat":     None,
        "origin_lon":     None,
        "dest_city":      m_del.group(1).strip() if m_del else None,
        "dest_state":     m_del.group(2).strip() if m_del else None,
        "dest_lat":       None,
        "dest_lon":       None,
        "pickup_start":   m_pu.group(3) if m_pu else None,
        "pickup_end":     m_pu.group(4) if m_pu else None,
        "delivery_start": m_del.group(3) if m_del else None,
        "delivery_end":   m_del.group(4) if m_del else None,
        "equipment_type": m_eq.group(1).strip() if m_eq else "Dry Van",
        "commodity":      m_eq.group(4).strip() if m_eq else None,
        "weight_lbs":     int(m_eq.group(3).replace(",", "")) if m_eq else None,
        "distance_miles": float(m_eq.group(2).replace(",", "")) if m_eq else None,
        "rate_usd":       float(m_pay.group(1).replace(",", "")) if m_pay else None,
        "rate_per_mile":  float(m_pay.group(2)) if m_pay else None,
        "requirements":   m_nt.group(1).strip() if m_nt else None,
    }


# Format C example:
# L-0000003 | Baltimore,MD -> Charlotte,NC
#   pu 2026-06-17 10:00  del 2026-06-18 07:00
#   Reefer 471mi wt 40400 comm=Meat & poultry
#   rate $1,150 rpm 2.44 req[None]

_C_LOAD_ID   = re.compile(r'^(L-\d+)\s*\|')
_C_HEADER    = re.compile(r'^L-\d+\s*\|\s*([^,]+),([A-Z]{2})\s*->\s*([^,\n]+),([A-Z]{2})', re.MULTILINE)
_C_TIMES     = re.compile(r'pu\s+([\d\-]+ [\d:]+)\s+del\s+([\d\-]+ [\d:]+)')
_C_EQUIP     = re.compile(r'(Dry Van|Flatbed|Reefer|Step Deck|Power Only)\s+([\d,]+)mi\s+wt\s+([\d,]+)\s+comm=(.+?)(?:\s+rate|$)', re.IGNORECASE)
_C_RATE      = re.compile(r'rate\s+\$([\d,]+)\s+rpm\s+([\d.]+)')
_C_REQ       = re.compile(r'req\[([^\]]+)\]')


def _parse_format_c(block: str) -> Optional[dict]:
    m_id  = _C_LOAD_ID.search(block)
    if not m_id:
        return None
    m_hdr = _C_HEADER.search(block)
    m_tm  = _C_TIMES.search(block)
    m_eq  = _C_EQUIP.search(block)
    m_rt  = _C_RATE.search(block)
    m_req = _C_REQ.search(block)
    return {
        "load_id":        m_id.group(1),
        "origin_city":    m_hdr.group(1).strip() if m_hdr else None,
        "origin_state":   m_hdr.group(2).strip() if m_hdr else None,
        "origin_lat":     None,
        "origin_lon":     None,
        "dest_city":      m_hdr.group(3).strip() if m_hdr else None,
        "dest_state":     m_hdr.group(4).strip() if m_hdr else None,
        "dest_lat":       None,
        "dest_lon":       None,
        "pickup_start":   m_tm.group(1) if m_tm else None,
        "pickup_end":     None,
        "delivery_start": m_tm.group(2) if m_tm else None,
        "delivery_end":   None,
        "equipment_type": m_eq.group(1).strip().title() if m_eq else "Dry Van",
        "commodity":      m_eq.group(4).strip() if m_eq else None,
        "weight_lbs":     int(m_eq.group(3).replace(",", "")) if m_eq else None,
        "distance_miles": float(m_eq.group(2).replace(",", "")) if m_eq else None,
        "rate_usd":       float(m_rt.group(1).replace(",", "")) if m_rt else None,
        "rate_per_mile":  float(m_rt.group(2)) if m_rt else None,
        "requirements":   m_req.group(1).strip() if m_req else None,
    }


# --- dispatch ---------------------------------------------------------------

def _parse_block(block: str) -> Optional[dict]:
    first = block.lstrip().split('\n', 1)[0]
    if first.startswith("LOAD "):
        return _parse_format_a(block)
    if first.startswith("==="):
        return _parse_format_b(block)
    return _parse_format_c(block)


# --- file / directory entry points ------------------------------------------

def parse_file(filepath: str) -> list[dict]:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        logger.warning("Cannot read %s: %s", filepath, exc)
        return []
    blocks = _split_blocks(text)
    records = []
    for block in blocks:
        rec = _parse_block(block)
        if rec and rec.get("origin_city") and rec.get("dest_city"):
            records.append(rec)
    return records


def parse_all_files(data_dir: Optional[str] = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data_dir = os.path.abspath(data_dir)

    txt_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True))
    if not txt_files:
        logger.warning("No .txt files found in %s — returning synthetic demo data", data_dir)
        return _synthetic_demo_data()

    all_records: list[dict] = []
    for path in txt_files:
        records = parse_file(path)
        logger.info("Parsed %d loads from %s", len(records), os.path.basename(path))
        all_records.extend(records)

    if not all_records:
        logger.warning("All files parsed to 0 records — returning synthetic demo data")
        return _synthetic_demo_data()

    df = pd.DataFrame(all_records)
    df = _fill_coordinates(df)
    df = _coerce_types(df)
    df = df[df["origin_city"].notna() & df["dest_city"].notna()].reset_index(drop=True)
    # Compatibility aliases used by dashboard.py and agent.py
    df["miles"] = df["distance_miles"]
    df["gross_rate"] = df["rate_usd"]
    df["equipment"] = df["equipment_type"]
    df["pickup_date"] = df["pickup_start"]
    logger.info("Total loads parsed: %d", len(df))
    return df


# --- helpers ----------------------------------------------------------------

def _fill_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    for city_col, state_col, lat_col, lon_col in [
        ("origin_city", "origin_state", "origin_lat", "origin_lon"),
        ("dest_city",   "dest_state",   "dest_lat",   "dest_lon"),
    ]:
        missing = df[lat_col].isna()
        if missing.any():
            coords = df[missing].apply(
                lambda r, c=city_col, s=state_col: lookup_coords(str(r[c]), str(r[s])) or (None, None),
                axis=1,
            )
            df.loc[missing, lat_col] = coords.apply(lambda x: x[0])
            df.loc[missing, lon_col] = coords.apply(lambda x: x[1])
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["distance_miles", "rate_usd", "rate_per_mile", "origin_lat", "origin_lon", "dest_lat", "dest_lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["weight_lbs"] = pd.to_numeric(df["weight_lbs"], errors="coerce").astype("Int64")
    for col in ["origin_city", "origin_state", "dest_city", "dest_state", "equipment_type", "commodity"]:
        df[col] = df[col].astype(str).str.strip()
    return df


# --- synthetic fallback (used only if no text files exist) ------------------

_DEMO_LANES = [
    ("Atlanta","GA","Fort Worth","TX",908,1700,"Dry Van"),
    ("Chicago","IL","Dallas","TX",925,1850,"Dry Van"),
    ("Los Angeles","CA","Phoenix","AZ",372,900,"Reefer"),
    ("Dallas","TX","Nashville","TN",675,1350,"Flatbed"),
    ("Miami","FL","Atlanta","GA",662,1400,"Dry Van"),
    ("Houston","TX","Chicago","IL",1090,2100,"Dry Van"),
    ("Seattle","WA","Denver","CO",1320,2400,"Flatbed"),
    ("Philadelphia","PA","Charlotte","NC",482,1050,"Dry Van"),
    ("Kansas City","MO","Indianapolis","IN",480,980,"Reefer"),
    ("Detroit","MI","Columbus","OH",174,520,"Dry Van"),
]


def _synthetic_demo_data() -> pd.DataFrame:
    rng = random.Random(42)
    base = datetime.date(2026, 6, 15)
    rows = []
    for i in range(600):
        oc, os_, dc, ds, mi, rate, eq = _DEMO_LANES[i % len(_DEMO_LANES)]
        rows.append({
            "load_id": f"DEMO{i:06d}",
            "origin_city": oc, "origin_state": os_,
            "origin_lat": None, "origin_lon": None,
            "dest_city": dc, "dest_state": ds,
            "dest_lat": None, "dest_lon": None,
            "pickup_start": base.isoformat(), "pickup_end": None,
            "delivery_start": None, "delivery_end": None,
            "equipment_type": eq,
            "commodity": "General Freight",
            "weight_lbs": 40000,
            "distance_miles": round(mi * rng.uniform(0.96, 1.04), 1),
            "rate_usd": round(rate * rng.uniform(0.85, 1.15), 2),
            "rate_per_mile": None,
            "requirements": "None",
        })
    df = pd.DataFrame(rows)
    df = _fill_coordinates(df)
    df["miles"] = df["distance_miles"]
    df["gross_rate"] = df["rate_usd"]
    df["equipment"] = df["equipment_type"]
    df["pickup_date"] = df["pickup_start"]
    return df
