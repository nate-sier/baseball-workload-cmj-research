"""Construct athlete-day research records using only Python Reports tabs.

No SQL connection, external file paths, or Streamlit imports.
"""
from __future__ import annotations

import re
import unicodedata
import numpy as np
import pandas as pd

YEAR = 2026
WINDOWS = (1, 3, 7, 14, 21, 28)
LOADS = ("max_effort_distance_covered_yards", "distance_covered_yards", "max_effort_runs", "games_played")
JUMP_METRICS = {
    "Concentric impulse (N·s)": "ci_ns",
    "CI first 100 ms (N·s)": "ci_100ms_ns",
    "mRSI": "mrsi",
    "Relative CI (N·s/kg)": "ci_rel_ns_kg",
    "Peak power (W)": "peak_power_w",
    "Relative peak power (W/kg)": "peak_power_rel_wkg",
}
LOAD_LABELS = {
    "Max-effort distance (yd)": "max_effort_distance_covered_yards",
    "Total distance (yd)": "distance_covered_yards",
    "Max-effort runs": "max_effort_runs",
    "Games played": "games_played",
}
ALIASES = {
    "ci_ns": ["Concentric Impulse [N s]", "Concentric Impulse (N s)", "Concentric Impulse", "ci_ns"],
    "ci_100ms_ns": ["Concentric Impulse 100 ms [N s]", "Concentric Impulse 100ms [N s]", "Concentric Impulse - 100ms [N s]", "CI100", "ci_100ms_ns"],
    "mrsi": ["mRSI", "RSI Modified (Imp-Mom)", "RSI-modified (Imp-Mom)", "RSI Modified", "Modified RSI", "mrsi"],
    "ci_rel_ns_kg": ["Concentric Impulse (Abs) / BM [N s/kg]", "Relative Concentric Impulse [N s/kg]", "ci_rel_ns_kg"],
    "peak_power_w": ["Peak Power [W]", "Peak Power (W)", "peak_power_w"],
    "peak_power_rel_wkg": ["Peak Power / BM [W/kg]", "Relative Peak Power [W/kg]", "peak_power_rel_wkg"],
}


def col(df: pd.DataFrame, *aliases: str, required: bool = False, context: str = "") -> str | None:
    candidates = {re.sub(r"[^a-z0-9]", "", str(c).casefold()): c for c in df.columns}
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.casefold())
        if key in candidates:
            return candidates[key]
    if required:
        raise ValueError(f"Missing {context} column ({', '.join(aliases)}). Available: {list(df.columns)[:20]}")
    return None


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if "," in s:
        a, b = s.split(",", 1)
        if a.strip() and b.strip():
            s = b.strip() + " " + a.strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9 ]", "", s.casefold()).split())


def _date(values: pd.Series) -> pd.Series:
    # Date values come from Google Sheets, sometimes with a time component.
    out = pd.to_datetime(values.replace("", np.nan), errors="coerce", format="mixed")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.replace({"": np.nan, "-": np.nan}), errors="coerce")


def build_snapshot(pp_raw: pd.DataFrame, jumps_raw: pd.DataFrame, *, name_map: dict | None = None,
                   blanks_are_zero: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Return (eligible snapshot, audit, within-session replicate trial data).

    All windows are calendar-day sums of recorded game load from days [test-W, test-1].
    Day of assessment is excluded. Full 28-day observation-history run-in applies
    to all windows; missing days between first/last games count as zero *recorded*
    game workload, which may differ from actual physical workload.
    """
    mapping = {normalize_name(a): normalize_name(b) for a, b in (name_map or {}).items()}
    pp_required = ["batter", "game_date", "mlbam_game_pk", *LOADS[:3]]
    cols = {key: col(pp_raw, key, required=True, context="PP_Sprint") for key in pp_required}
    pp = pp_raw.copy()
    pp["game_date"] = _date(pp[cols["game_date"]])
    pp = pp[pp["game_date"].dt.year.eq(YEAR)].copy()
    pp["batter"] = pp[cols["batter"]].fillna("").astype(str).str.strip()
    pp["player_key"] = pp["batter"].map(normalize_name).replace(mapping)
    pp["game_id"] = (pp[cols["mlbam_game_pk"]].fillna("").astype(str).str.strip()
                     .str.replace(r"\.0$", "", regex=True))
    pp = pp[pp["player_key"].ne("") & pp["game_date"].notna()].copy()
    if pp.empty:
        raise ValueError("No valid 2026 rows found in PP_Sprint")
    if pp["game_id"].isin(["", "nan", "None"]).any():
        raise ValueError("PP_Sprint has missing MLBAM game IDs; correct these before calculating games")
    blank_counts = {}
    for key in LOADS[:3]:
        pp[key] = _numeric(pp[cols[key]])
        blank_counts[key] = int(pp[key].isna().sum())
        if blank_counts[key]:
            if not blanks_are_zero:
                raise ValueError(f"PP_Sprint has {blank_counts[key]} missing/non-numeric '{key}' values. "
                                 "Correct them in the Sheet or explicitly enable 'Missing load values mean zero'.")
            pp[key] = pp[key].fillna(0)
        if pp[key].lt(0).any():
            raise ValueError(f"Negative game workload in {key}")
    game_key = ["player_key", "game_date", "game_id"]
    clashes = pp.groupby(game_key)[list(LOADS[:3])].nunique(dropna=False)
    if (clashes > 1).any(axis=1).any():
        raise ValueError("Conflicting duplicate player/game rows in PP_Sprint; resolve before analysis")
    n_dupes = int(pp.duplicated(game_key).sum())
    pp = pp.drop_duplicates(game_key)
    daily = (pp.groupby(["player_key", "game_date"], as_index=False)
             .agg(batter=("batter", "first"),
                  max_effort_distance_covered_yards=("max_effort_distance_covered_yards", "sum"),
                  distance_covered_yards=("distance_covered_yards", "sum"),
                  max_effort_runs=("max_effort_runs", "sum"),
                  games_played=("game_id", "nunique")))

    name_c = col(jumps_raw, "Athlete", "Name", "athlete_name", required=True, context="Jump Data")
    date_c = col(jumps_raw, "Date", "jump_date", "Test Date", required=True, context="Jump Data")
    found = {k: col(jumps_raw, *names) for k, names in ALIASES.items()}
    if not found["ci_ns"]:
        raise ValueError("Jump Data does not have the expected concentric impulse column")
    jump = jumps_raw.copy()
    jump["Athlete"] = jump[name_c].fillna("").astype(str).str.strip()
    jump["player_key"] = jump["Athlete"].map(normalize_name)
    jump["jump_date"] = _date(jump[date_c])
    jump = jump[jump["jump_date"].dt.year.eq(YEAR) & jump["player_key"].ne("")].copy()
    present = []
    for metric, source in found.items():
        if source is not None:
            jump[metric] = _numeric(jump[source])
            if jump[metric].notna().any():
                present.append(metric)
    if "ci_ns" not in present:
        raise ValueError("No numeric 2026 concentric impulse data found in Jump Data")
    if jump.empty:
        raise ValueError("No 2026 athlete jump records found")
    replicates = jump[["player_key", "jump_date", *present]].copy()
    # One observation per athlete per assessment date: mean of all valid trials.
    jd = (jump.groupby(["player_key", "jump_date"], as_index=False)
          .agg(**{metric: (metric, "mean") for metric in present}, Athlete=("Athlete", "first")))
    jd["days_since_prior_jump"] = (jd["jump_date"] - jd.groupby("player_key")["jump_date"].shift()).dt.days
    # Define events on the complete jump sequence BEFORE dropping first 28 days.
    for metric in ["ci_ns", "ci_100ms_ns", "mrsi"]:
        if metric not in present:
            continue
        group = jd.groupby("player_key")[metric]
        prior1 = group.shift(1)
        prior3 = jd.groupby("player_key")[metric].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
        for label, baseline in (("prior1", prior1), ("prior3", prior3)):
            jd[f"drop_pct_{label}_{metric}"] = np.where(
                baseline.gt(0), 100 * (jd[metric] - baseline) / baseline, np.nan)

    pp_athletes = set(daily["player_key"])
    jump_athletes = set(jd["player_key"])
    overlap = pp_athletes & jump_athletes
    if not overlap:
        raise ValueError("No athlete names matched between PP_Sprint.batter and Jump Data.Athlete")
    unmatched = pd.concat([
        daily[~daily["player_key"].isin(overlap)][["batter"]].drop_duplicates().rename(columns={"batter": "name"}).assign(source="PP_Sprint"),
        jd[~jd["player_key"].isin(overlap)][["Athlete"]].drop_duplicates().rename(columns={"Athlete": "name"}).assign(source="Jump Data"),
    ], ignore_index=True)
    # Only consider dates from first game + 28 days through the day after last
    # observed game. No pretend-zero rows outside observed team/game coverage.
    grids = []
    for player_key, part in daily[daily["player_key"].isin(overlap)].groupby("player_key"):
        start = part["game_date"].min()
        stop = part["game_date"].max() + pd.Timedelta(days=1)
        dates = pd.date_range(start, stop, freq="D")
        grid = pd.DataFrame({"player_key": player_key, "jump_date": dates})
        part = part.rename(columns={"game_date": "jump_date"})
        grid = grid.merge(part[["jump_date", *LOADS]], on="jump_date", how="left", validate="one_to_one")
        grid[list(LOADS)] = grid[list(LOADS)].fillna(0)
        for metric in LOADS:
            for days in WINDOWS:
                grid[f"{metric}_{days}d"] = grid[metric].shift(1).rolling(days, min_periods=days).sum()
        grids.append(grid[["player_key", "jump_date", *[f"{m}_{w}d" for m in LOADS for w in WINDOWS]]])
    lookup = pd.concat(grids, ignore_index=True)
    before = int(len(jd[jd["player_key"].isin(overlap)]))
    result = jd.merge(lookup, on=["player_key", "jump_date"], how="inner", validate="one_to_one")
    result = result.dropna(subset=[f"{m}_28d" for m in LOADS]).sort_values(["player_key", "jump_date"]).reset_index(drop=True)
    if result.empty:
        raise ValueError("No jump assessments have 28 full prior days of recorded game history. Check date coverage.")
    # Do not expose raw private names via audit; show counts only.
    audit = {
        "source_games": len(pp), "source_jump_trials": len(jump), "eligible_athletes": result["player_key"].nunique(),
        "eligible_athlete_days": len(result), "excluded_player_days": before - len(result),
        "unmatched_pp_names": len(unmatched[unmatched["source"].eq("PP_Sprint")]),
        "unmatched_jump_names": len(unmatched[unmatched["source"].eq("Jump Data")]),
        "exact_duplicate_games_removed": n_dupes,
        "blank_game_metrics_treated_as_zero": blank_counts if blanks_are_zero else {},
        "found_jump_metrics": present,
    }
    return result, audit, replicates


def within_trial_typical_error(replicates: pd.DataFrame, metric: str) -> dict:
    """Pooled within-athlete-date trial SD; NOT day-to-day test-retest error."""
    if metric not in replicates:
        return {"te": np.nan, "groups": 0, "df": 0}
    d = replicates[["player_key", "jump_date", metric]].dropna()
    g = d.groupby(["player_key", "jump_date"])[metric]
    counts = g.transform("size")
    d = d[counts >= 2].copy()
    if d.empty:
        return {"te": np.nan, "groups": 0, "df": 0}
    g2 = d.groupby(["player_key", "jump_date"])[metric]
    residual = d[metric] - g2.transform("mean")
    degrees = int(len(d) - g2.ngroups)
    return {"te": float(np.sqrt((residual**2).sum() / degrees)) if degrees > 0 else np.nan,
            "groups": int(g2.ngroups), "df": degrees}
