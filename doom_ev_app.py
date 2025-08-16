
# doom_ev_app.py
# Streamlit UI for Doom of Mokhaiotl EV & Bank-Threshold Calculator
#
# - Live prices via OSRS Wiki GE API
# - Exact delve quantity multiplier rule
# - Per-level and per-run EVs (uniques / commons / both)
# - Cumulative unique chance across a run
# - Bank-now threshold after your chosen end level, given next-level success chance
# - NEW: Runs-per-hour input and hourly EV / unique odds
#
# Usage:
#   streamlit run doom_ev_app.py
#
# Requires: streamlit, requests, pandas

import math
import re
from typing import Dict, List, Tuple, Optional

import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Doom EV & Bank Threshold (OSRS)", layout="wide")

# -----------------------------
# OSRS Wiki GE API endpoints
# -----------------------------
MAPPING_URL = "https://prices.runescape.wiki/api/v1/osrs/mapping"
LATEST_URL  = "https://prices.runescape.wiki/api/v1/osrs/latest"
USER_AGENT  = "doom-ev-tool/1.1 (contact: put-your-email-or-discord-here)"  # Please customize

# -----------------------------
# Delve mechanics/config
# -----------------------------

# Quantity multipliers by delve level. 9+ uses the same as 9.
Q_MULT = {1: -0.50, 2: -0.35, 3: 0.00, 4: 0.05, 5: 0.10, 6: 0.12, 7: 0.14, 8: 0.17, 9: 0.20}
def qmult(level: int) -> float:
    return Q_MULT[9] if level >= 9 else Q_MULT[level]

# Guaranteed demon tears per level
GAR_TEARS = {1:0, 2:0, 3:50, 4:60, 5:70, 6:80, 7:90, 8:100, 9:100}
def guaranteed_tears(level: int) -> int:
    return GAR_TEARS[9] if level >= 9 else GAR_TEARS[level]

# Unique drop rates per level (per-item, as "1/x") — Dom pet excluded on purpose
UNIQUE_RATES = {
    "cloth":  {2:2500, 3:2000, 4:1350, 5:810, 6:765, 7:720, 8:630, 9:540},
    "eye":    {3:2000, 4:1350, 5:810, 6:765, 7:720, 8:630, 9:540},
    "treads": {4:1350, 5:810, 6:765, 7:720, 8:630, 9:540},
}
UNIQUE_ITEM_NAMES = {
    "cloth":  "Confliction gauntlets",  # value cloth turns into
    "eye":    "Eye of ayak (uncharged)",
    "treads": "Avernic treads",
}

# Regular (non-unique) drop table at "level 3 quantities" & base per-kill rarity.
COMMON_DROPS = [
    # Weapons & armour
    {"name": "Dragon med helm",      "prob": "5/105",  "q3": (1,1)},
    {"name": "Mystic earth staff",   "prob": "5/105",  "q3": (1,1)},
    {"name": "Rune pickaxe",         "prob": "5/105",  "q3": (1,3)},
    {"name": "Dragon platelegs",     "prob": "1/105",  "q3": (2,4)},

    # Runes & ammunition
    {"name": "Death rune",           "prob": "5/105",  "q3": (50,70)},
    {"name": "Chaos rune",           "prob": "5/105",  "q3": (50,70)},
    {"name": "Earth rune",           "prob": "5/105",  "q3": (500,1000)},
    {"name": "Fire rune",            "prob": "5/105",  "q3": (500,1000)},
    {"name": "Cannonball",           "prob": "5/105",  "q3": (200,600)},
    {"name": "Onyx bolts",           "prob": "5/105",  "q3": (5,15)},

    # Ores
    {"name": "Coal",                 "prob": "5/105",  "q3": (15,50)},
    {"name": "Gold ore",             "prob": "5/105",  "q3": (20,60)},
    {"name": "Runite ore",           "prob": "5/105",  "q3": (3,6)},

    # Seeds
    {"name": "Celastrus seed",       "prob": "3/105",  "q3": (1,1)},
    {"name": "Spirit seed",          "prob": "3/105",  "q3": (1,1)},  # Not sold
    {"name": "Ranarr seed",          "prob": "2/105",  "q3": (1,3)},

    # Resources
    {"name": "Aether catalyst",      "prob": "5/105",  "q3": (150,400)},
    {"name": "Dragon dart tip",      "prob": "5/105",  "q3": (30,90)},
    {"name": "Sun-kissed bones",     "prob": "5/105",  "q3": (25,75)},  # Not sold
    {"name": "Raw shark",            "prob": "3/105",  "q3": (20,35)},
    {"name": "Shark lure",           "prob": "2/105",  "q3": (40,70)},
    {"name": "Sunfire splinters",    "prob": "1/105",  "q3": (500,1500)},

    # Other
    {"name": "Demon tear",           "prob": "7/105",  "q3": (100,300)},
    {"name": "Mokhaiotl waystone",   "prob": "7/105",  "q3": (1,2)},    # may be Not sold
    {"name": "Tooth half of key (moon key)", "prob": "1/105", "q3": (1,1)},  # Not sold
    # Clue scroll (elite): tertiary and Not sold, ignore for EV
]

# Items considered Not sold → price 0 if GE price is missing
FORCE_ZERO_IF_MISSING = {
    "Spirit seed",
    "Sun-kissed bones",
    "Clue scroll (elite)",
    "Tooth half of key (moon key)",
}

# -----------------------------
# Helpers
# -----------------------------

def parse_prob(frac: str) -> float:
    num, den = frac.split("/")
    return float(num) / float(den)

def trunc_toward_zero(x: float) -> int:
    return int(x)  # trunc toward zero

def expected_qty_at_level(q3_range: Tuple[int, int], level: int) -> float:
    a, b = q3_range
    m = qmult(level)
    total = 0
    count = (b - a + 1)
    for q3 in range(a, b+1):
        qn = q3 + trunc_toward_zero(q3 * m)
        total += qn
    return total / count

@st.cache_data(ttl=60*5, show_spinner=False)
def fetch_mapping() -> List[dict]:
    r = requests.get(MAPPING_URL, headers={"User-Agent": USER_AGENT}, timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_latest_prices() -> dict:
    r = requests.get(LATEST_URL, headers={"User-Agent": USER_AGENT}, timeout=15)
    r.raise_for_status()
    return r.json()["data"]

def build_name_to_id(mapping: List[dict]) -> Dict[str, int]:
    out = {}
    for it in mapping:
        name = it.get("name", "").strip().lower()
        id_ = it.get("id")
        if name and id_ is not None:
            out[name] = id_
    return out

def price_picker(entry: dict, price_type: str) -> Optional[int]:
    hi = entry.get("high")
    lo = entry.get("low")
    if price_type == "high":
        return hi
    if price_type == "low":
        return lo
    # mid
    if hi is None and lo is None:
        return None
    if hi is None:
        return lo
    if lo is None:
        return hi
    return int(round((hi + lo) / 2))

def get_prices_for(names: List[str], price_type: str, overrides: Dict[str, int]) -> Dict[str, int]:
    """Return prices for given display names using OSRS GE API; apply overrides last."""
    try:
        mapping = fetch_mapping()
        name2id = build_name_to_id(mapping)
        latest = fetch_latest_prices()
    except Exception as e:
        st.error(f"Price API error: {e}")
        latest = {}
        name2id = {}

    out: Dict[str, int] = {}
    for name in names:
        price = None
        key = name.strip().lower()
        id_ = name2id.get(key)
        if id_ is not None and latest:
            entry = latest.get(str(id_))
            if entry:
                price = price_picker(entry, price_type)
        if price is None:
            # Default to 0 if missing or Not sold
            price = 0 if (name in FORCE_ZERO_IF_MISSING) else 0
        out[name] = int(price or 0)

    # Apply overrides (manual values win)
    for k, v in overrides.items():
        if v is not None and v > 0:
            out[k] = int(v)

    return out

def per_level_unique_any_prob(level: int) -> float:
    p = 0.0
    for key in ("cloth", "eye", "treads"):
        denom = UNIQUE_RATES[key].get(min(level, 9))
        if denom:
            p += 1.0 / float(denom)
    return p

def ev_uniques_for_level(level: int, prices: Dict[str, int], mode: str) -> float:
    if mode == "commons":
        return 0.0
    p_sum = 0.0
    for key, item_name in UNIQUE_ITEM_NAMES.items():
        denom = UNIQUE_RATES[key].get(min(level, 9))
        if denom:
            p = 1.0 / float(denom)
            p_sum += p * prices.get(item_name, 0)
    return p_sum

def ev_commons_for_level(level: int, prices: Dict[str, int], mode: str, include_tears: bool) -> float:
    # Commons EV excludes tears if include_tears is False; tears are *only* part of commons/both modes.
    if mode == "uniques":
        return 0.0
    ev = 0.0
    # One roll on the regular table
    for it in COMMON_DROPS:
        p = parse_prob(it["prob"])
        q_exp = expected_qty_at_level(it["q3"], level)
        ev += p * q_exp * prices.get(it["name"], 0)
    if include_tears:
        ev += guaranteed_tears(level) * prices.get("Demon tear", 0)
    return ev

def ev_level_total(level: int, prices: Dict[str, int], mode: str, include_tears: bool) -> Dict[str, float]:
    eu = ev_uniques_for_level(level, prices, mode)
    ec = ev_commons_for_level(level, prices, mode, include_tears)
    return {"uniques": eu, "commons": ec, "total": eu + ec}

def cumulative_unique_prob_over_run(start_level: int, end_level: int) -> float:
    fail = 1.0
    for L in range(start_level, end_level + 1):
        p = per_level_unique_any_prob(L)
        fail *= (1.0 - p)
    return 1.0 - fail

def expected_uniques_per_run(start_level: int, end_level: int) -> float:
    """Linearity of expectation: sum of per-level p(any unique on that level)."""
    total = 0.0
    for L in range(start_level, end_level + 1):
        total += per_level_unique_any_prob(L)
    return total

def expected_runs_to_unique(start_level: int, end_level: int) -> Optional[float]:
    p = cumulative_unique_prob_over_run(start_level, end_level)
    if p <= 0:
        return None
    return 1.0 / p

def bank_threshold_after_end(end_level: int, prices: Dict[str, int], mode: str,
                             include_tears: bool, success_next: float, death_fee: int) -> float:
    d = 1.0 - success_next
    if d <= 0:
        return float("inf")
    ev_next = ev_level_total(end_level + 1, prices, mode, include_tears)["total"]
    return ((1.0 - d) / d) * ev_next - death_fee

def gp(x: float) -> str:
    return f"{x:,.0f}"

# -----------------------------
# UI
# -----------------------------

st.title("Doom of Mokhaiotl — EV & Bank Threshold Calculator")
st.caption("Live OSRS Wiki GE prices • Cloth valued as Confliction gauntlets • Dom pet excluded")

with st.sidebar:
    st.header("Configuration")
    cols = st.columns(2)
    with cols[0]:
        start_level = st.number_input("Start level", min_value=1, max_value=20, value=1, step=1)
    with cols[1]:
        end_level = st.number_input("End level", min_value=1, max_value=20, value=7, step=1)
    if end_level < start_level:
        st.warning("End level must be ≥ start level. Adjusting end= start.")
        end_level = start_level

    st.markdown("**Note:** For 9+, rates/multipliers are treated as level 9.")

    mode = st.selectbox("EV mode", options=["both", "uniques", "commons"],
                        format_func=lambda s: {"both":"Both (uniques + commons)",
                                               "uniques":"Uniques only",
                                               "commons":"Non-uniques only"}[s],
                        index=0)

    include_tears = st.checkbox("Include guaranteed Demon tears (commons/both)", value=True,
                                help="If unchecked, guaranteed tears are excluded from EV. Tears are not included when mode = 'uniques'.")

    price_type = st.selectbox("Price type", options=["low", "mid", "high"], index=0,
                              help="Use low, mid (average), or high from OSRS GE latest.")
    death_fee = st.number_input("Death fee (gp)", min_value=0, value=25_000, step=1_000)

    st.subheader("Next-level success chance")
    success_pct = st.slider(f"Success chance for level {end_level+1} if you continue", 0, 100, 50, 1)
    success_next = success_pct / 100.0

    st.subheader("Throughput")
    runs_per_hour = st.number_input("Runs per hour (completed full runs of your selected range)", min_value=1, max_value=60, value=2, step=1)

    st.subheader("Overrides (optional)")
    st.caption("Use these if an item is Not sold / missing on GE. Set to 0 to disable override.")
    override_confliction = st.number_input("Override price: Confliction gauntlets (gp)", min_value=0, value=0, step=10_000)
    override_demon_tear = st.number_input("Override price: Demon tear (per tear, gp)", min_value=0, value=0, step=100,
                                          help="Guaranteed tears use this per-item price if set.")

# Build price map
needed_names = set([it["name"] for it in COMMON_DROPS])
needed_names.update(UNIQUE_ITEM_NAMES.values())
needed_names.add("Demon tear")

overrides = {}
if override_confliction > 0:
    overrides["Confliction gauntlets"] = override_confliction
if override_demon_tear > 0:
    overrides["Demon tear"] = override_demon_tear

prices = get_prices_for(sorted(needed_names), price_type, overrides)

# Display current unique prices
st.subheader("Current unique prices")
price_rows = []
for key, nm in UNIQUE_ITEM_NAMES.items():
    price_rows.append({"Item": nm, "Price (gp)": prices.get(nm, 0)})
st.table(pd.DataFrame(price_rows))

# Per-level table
rows = []
cum_fail = 1.0
ev_tot_uni = 0.0
ev_tot_com = 0.0

for L in range(int(start_level), int(end_level) + 1):
    evs = ev_level_total(L, prices, mode, include_tears)
    pL = per_level_unique_any_prob(L)
    cum_fail *= (1.0 - pL)
    cum_p = 1.0 - cum_fail
    rows.append({
        "Level": L,
        "EV_uniques (gp)": evs["uniques"],
        "EV_commons (gp)": evs["commons"],
        "EV_total (gp)": evs["total"],
        "p_unique(L)": pL,
        "cum_p_unique(≤L)": cum_p,
    })
    ev_tot_uni += evs["uniques"]
    ev_tot_com += evs["commons"]

df = pd.DataFrame(rows)
# Pretty formatting on display
df_show = df.copy()
df_show["EV_uniques (gp)"] = df_show["EV_uniques (gp)"].map(lambda v: f"{int(round(v)):,}")
df_show["EV_commons (gp)"] = df_show["EV_commons (gp)"].map(lambda v: f"{int(round(v)):,}")
df_show["EV_total (gp)"] = df_show["EV_total (gp)"].map(lambda v: f"{int(round(v)):,}")
df_show["p_unique(L)"] = df_show["p_unique(L)"].map(lambda v: f"{v:.4%}")
df_show["cum_p_unique(≤L)"] = df_show["cum_p_unique(≤L)"].map(lambda v: f"{v:.4%}")

st.subheader("Per-level EV & unique probabilities")
st.dataframe(df_show, use_container_width=True)

# Run totals
ev_run_total = ev_tot_uni + ev_tot_com
p_unique_run = cumulative_unique_prob_over_run(int(start_level), int(end_level))
exp_uniques_run = expected_uniques_per_run(int(start_level), int(end_level))
runs_to_unique = expected_runs_to_unique(int(start_level), int(end_level))

colA, colB, colC = st.columns(3)
with colA:
    st.metric("EV uniques (per run)", f"{int(round(ev_tot_uni)):,} gp")
with colB:
    st.metric("EV commons (per run)", f"{int(round(ev_tot_com)):,} gp")
with colC:
    st.metric("EV total (per run)", f"{int(round(ev_run_total)):,} gp")

colD, colE, colF = st.columns(3)
with colD:
    st.metric("P(≥1 unique per run)", f"{p_unique_run:.2%}")
with colE:
    st.metric("Expected uniques per run", f"{exp_uniques_run:.4f}")
with colF:
    st.metric("Expected runs to a unique", "N/A" if runs_to_unique is None else f"{runs_to_unique:.2f}")

# Hourly stats
st.subheader("Hourly stats (based on runs per hour)")
ev_uni_hour = ev_tot_uni * runs_per_hour
ev_com_hour = ev_tot_com * runs_per_hour
ev_total_hour = ev_run_total * runs_per_hour
p_unique_hour = 1.0 - (1.0 - p_unique_run) ** runs_per_hour
exp_uniques_hour = exp_uniques_run * runs_per_hour

colH1, colH2, colH3 = st.columns(3)
with colH1:
    st.metric("EV uniques per hour", f"{int(round(ev_uni_hour)):,} gp/h")
with colH2:
    st.metric("EV commons per hour", f"{int(round(ev_com_hour)):,} gp/h")
with colH3:
    st.metric("EV total per hour", f"{int(round(ev_total_hour)):,} gp/h")

colH4, colH5 = st.columns(2)
with colH4:
    st.metric("P(≥1 unique in an hour)", f"{p_unique_hour:.2%}")
with colH5:
    st.metric("Expected uniques per hour", f"{exp_uniques_hour:.4f}")

# Bank threshold
v_star = bank_threshold_after_end(int(end_level), prices, mode, include_tears, success_next, int(death_fee))
ev_next = ev_level_total(int(end_level) + 1, prices, mode, include_tears)["total"]

st.subheader("Decision: Bank now vs. attempt next level once")
st.write(f"**Next level:** {int(end_level)+1}  |  **Success chance:** {success_pct}%  "
         f"(death chance {100 - success_pct}%)  |  **EV(next level):** {int(round(ev_next)):,} gp  "
         f"|  **Death fee:** {int(death_fee):,} gp")

if math.isfinite(v_star):
    if v_star < 0:
        st.success(f"Bank threshold V*: {int(round(v_star)):,} gp → negative threshold ⇒ Always **continue** (EV-positive).")
    else:
        st.info(f"**Bank if your current bag ≥ {int(round(v_star)):,} gp.**  Otherwise, continue to the next level.")
else:
    if success_next >= 1.0:
        st.success("With 100% success, always continue (threshold is ∞).")
    else:
        st.warning("With 0% success, always bank (threshold = −death fee).")

with st.expander("Price details (for troubleshooting)", expanded=False):
    st.write("Item prices used (after overrides):")
    price_items = sorted(list(needed_names))
    data = [{"Item": nm, "Price (gp)": prices.get(nm, 0)} for nm in price_items]
    st.table(pd.DataFrame(data))

st.caption("Notes: Cloth is valued as Confliction gauntlets. Dom pet excluded. "
           "Items missing/Not sold on GE are treated as 0 unless overridden. "
           "9+ depths reuse level‑9 rates. Uniques per run/hour are expectations; multiple uniques can occur across levels in a run.")
