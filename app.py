"""Professional baseball: game workload and subsequent CMJ | Streamlit entry point.

Deployment main file: app.py (at the GitHub repository ROOT).
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_pipeline import (YEAR, WINDOWS, LOAD_LABELS, JUMP_METRICS,
                           build_snapshot, within_trial_typical_error)
from analysis import fit_window, analyze_drops

st.set_page_config(page_title="Baseball game workload & CMJ", layout="wide")
st.title("Game workload & countermovement jump performance")
st.caption("2026 professional baseball · Athlete-specific random intercepts and workload slopes · Python Reports")


def settings():
    try:
        return dict(st.secrets)
    except Exception:
        return {}


def sheets_config():
    secret = settings()
    sid = os.getenv("PYTHON_REPORTS_SHEET_ID", "").strip() or str(secret.get("sheet_id", "")).strip()
    import re
    match = re.search(r"docs\.google\.com/spreadsheets/d/([\w-]+)", sid)
    if match:
        sid = match.group(1)
    account = None
    for key in ("gcp_service_account", "google_service_account"):
        if key in secret:
            try:
                account = dict(secret[key])
            except (ValueError, TypeError):
                pass
            if account and account.get("private_key"):
                break
    if not account:
        filepath = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        if not filepath:
            filepath = str(Path.home() / "Desktop" / "service_account.json")
        path = Path(filepath).expanduser()
        if path.is_file():
            account = json.loads(path.read_text(encoding="utf-8"))
    return sid, account


def pull_google(sid: str, account: dict):
    import gspread
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_info(
        account, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
    sheet = gspread.authorize(creds).open_by_key(sid)
    pp = pd.DataFrame(sheet.worksheet("PP_Sprint").get_all_records())
    jumps = pd.DataFrame(sheet.worksheet("Jump Data").get_all_records())
    return pp, jumps


def read_workbook(data: bytes):
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=["PP_Sprint", "Jump Data"], engine="openpyxl")
    return sheets["PP_Sprint"], sheets["Jump Data"]


def clear_calculations():
    for key in ("processed", "model_results", "analysis_signature"):
        st.session_state.pop(key, None)


st.sidebar.header("Data source")
sheet_id, service_account = sheets_config()
ready_google = bool(sheet_id and service_account)
source = st.sidebar.radio("Load from", ["Google Sheets", "Excel workbook"],
                          index=0 if ready_google else 1)
name_map_file = st.sidebar.file_uploader("Optional name matching CSV (pp_batter,jump_athlete)", type="csv")
allow_blanks = st.sidebar.checkbox("Missing game-load cells actually mean zero", value=False,
                                   help="Only enable if blanks really indicate zero recorded workload.")
name_map = {}
if name_map_file:
    from data_pipeline import normalize_name
    try:
        mapped = pd.read_csv(name_map_file).fillna("")
        if not {"pp_batter", "jump_athlete"}.issubset(mapped):
            raise ValueError("Expected columns: pp_batter,jump_athlete")
        name_map = {normalize_name(a): normalize_name(b)
                    for a, b in zip(mapped["pp_batter"], mapped["jump_athlete"])
                    if normalize_name(a) and normalize_name(b)}
    except Exception as exc:
        st.sidebar.error(str(exc))
        st.stop()

if source == "Google Sheets":
    if not ready_google:
        st.sidebar.warning("Set sheet_id and [gcp_service_account] in Streamlit Secrets.")
    uploaded = None
else:
    uploaded = st.sidebar.file_uploader("Python Reports (.xlsx)", type="xlsx")

load_clicked = st.sidebar.button("Load / refresh source data", type="primary", use_container_width=True)
if load_clicked:
    clear_calculations()
    try:
        with st.spinner("Reading Python Reports and constructing eligible player-day observations…"):
            if source == "Google Sheets":
                if not ready_google:
                    raise ValueError("Google Sheets is not configured; see README and Streamlit Cloud Secrets")
                pp, jumps = pull_google(sheet_id, service_account)
            else:
                if uploaded is None:
                    raise ValueError("Choose the Python Reports Excel workbook")
                pp, jumps = read_workbook(uploaded.getvalue())
            snapshot, audit, trials = build_snapshot(pp, jumps, name_map=name_map,
                                                    blanks_are_zero=allow_blanks)
        st.session_state["processed"] = (snapshot, audit, trials)
        st.session_state["source_label"] = source
    except Exception as exc:
        # Show exception type and sanitized human-readable content, not secret/key material.
        st.error(f"Data loading failed ({type(exc).__name__}): {str(exc)[:450]}")

if "processed" not in st.session_state:
    st.info("Start by clicking **Load / refresh source data** in the sidebar. "
            "The app reads the Python Reports Google Sheet directly once Streamlit Secrets are configured.")
    st.markdown("**Required worksheets:** `PP_Sprint` and `Jump Data`. No SQL or snapshot export required.")
    st.stop()

snapshot, audit, trials = st.session_state["processed"]
main1, main2, main3 = st.tabs(["Mixed-effects models", "Acute ≥5% drops", "Data audit & methods"])
available = {label: metric for label, metric in JUMP_METRICS.items()
             if metric in snapshot and snapshot[metric].notna().any()}

with main1:
    st.subheader("Raw jump outcomes | Athlete random intercept + workload slope")
    c1, c2, c3 = st.columns([1.1, 1.1, 0.8])
    with c1:
        metric_label = st.selectbox("Jump metric (absolute/raw value)", list(available), key="jump_metric")
        metric = available[metric_label]
    with c2:
        load_label = st.selectbox("Recorded in-game workload", list(LOAD_LABELS), key="load_metric")
        load = LOAD_LABELS[load_label]
    with c3:
        adjust_gap = st.checkbox("Adjust for days since prior jump", value=True,
                                 help="Removes first jump for each athlete and adjusts gap up to 30 days.")
    st.caption("Six cumulative calendar-day windows exclude test day. Within-athlete centering separates "
               "an athlete's workload fluctuations from their mean workload. Month is a fixed effect. "
               "Models are fitted by maximum likelihood (ML).")
    signature = (metric, load, adjust_gap, audit["eligible_athlete_days"], st.session_state.get("source_label"))
    if st.button("Fit all six mixed-effects models", type="primary", key="fit_models"):
        results = []
        progress = st.progress(0, text="Fitting random slopes…")
        for i, days in enumerate(WINDOWS):
            try:
                r = fit_window(snapshot, load, metric, days, adjust_gap=adjust_gap,
                               compare_intercept=True)
                r["Status"] = "WARNING" if r.get("Warnings") else "OK"
            except Exception as exc:
                r = {"Window (days)": days, "Status": f"FAILED: {type(exc).__name__}: {str(exc)[:200]}"}
            results.append(r)
            progress.progress((i + 1) / len(WINDOWS), text=f"Completed {i + 1} / {len(WINDOWS)} windows")
        progress.empty()
        trial_error = within_trial_typical_error(trials, metric)
        if np.isfinite(trial_error["te"]) and trial_error["te"] > 0:
            for row in results:
                effect = row.get("Effect (jump units / within-X SD)", np.nan)
                row["Effect / within-session trial SD"] = effect / trial_error["te"] if np.isfinite(effect) else np.nan
        st.session_state["model_results"] = pd.DataFrame(results)
        st.session_state["analysis_signature"] = signature
    if "model_results" in st.session_state and st.session_state.get("analysis_signature") == signature:
        report = st.session_state["model_results"]
        successes = report[report["Status"].isin(["OK", "WARNING"])].copy()
        st.markdown("#### Model estimates")
        if (report["Status"] == "WARNING").any():
            st.warning("One or more models emitted numerical/variance warnings despite converging. Review the Warnings column before interpreting their confidence intervals or p-values.")
        if len(successes):
            cols = ["Window (days)", "Athletes", "Jump observations",
                    "Effect (jump units / within-X SD)", "95% CI lower", "95% CI upper", "p-value",
                    "Standardized effect", "Within-X SD", "Within-Y SD", "Random-slope SD",
                    "Marginal R²", "Conditional R²", "Lag-1 residual r"]
            if "Effect / within-session trial SD" in successes:
                cols.append("Effect / within-session trial SD")
            show = successes[cols].copy()
            st.dataframe(show.style.format(precision=3), use_container_width=True, hide_index=True)
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=successes["Effect (jump units / within-X SD)"],
                y=successes["Window (days)"].astype(str) + " days", mode="markers",
                error_x=dict(type="data", symmetric=False,
                    array=(successes["95% CI upper"]-successes["Effect (jump units / within-X SD)"]).to_numpy(),
                    arrayminus=(successes["Effect (jump units / within-X SD)"]-successes["95% CI lower"]).to_numpy()),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            chart.add_vline(x=0, line_dash="dash")
            chart.update_layout(xaxis_title=f"{metric_label} change per pooled within-athlete workload SD",
                                yaxis_title="Workload lookback window", height=400)
            st.plotly_chart(chart, use_container_width=True)
            st.markdown("#### Model and residual checks")
            cols_extra = ["Window (days)", "Median athlete X SD", "Median athlete Y SD",
                          "Effect (% mean jump)", "Random-slope AIC", "Random-intercept AIC",
                          "AIC difference (slope - intercept)", "Residual pairs",
                          "Median residual pair gap (days)", "Converged", "Warnings"]
            st.dataframe(successes[cols_extra].style.format(precision=3),
                         hide_index=True, use_container_width=True)
            te = within_trial_typical_error(trials, metric)
            if np.isfinite(te["te"]):
                st.caption(f"Within-session replicate trial SD: {te['te']:.3f} raw jump units "
                           f"from {te['groups']} athlete-days ({te['df']} residual degrees of freedom). "
                           "This is NOT day-to-day test–retest typical error.")
            else:
                st.caption("Within-session typical error unavailable: no repeated trials on athlete-days.")
            st.download_button("Download model results (CSV)", report.to_csv(index=False).encode(),
                               file_name="mixed_effects_results_2026.csv", mime="text/csv")
        failed = report[report["Status"].str.startswith("FAILED", na=False)]
        if len(failed):
            st.warning(f"{len(failed)} model(s) failed; estimates were not substituted with another model.")
            st.dataframe(failed[["Window (days)", "Status"]], hide_index=True, use_container_width=True)
    else:
        st.info("Select your outcome and game workload, then click **Fit all six mixed-effects models**.")

with main2:
    st.subheader("Workload preceding ≥5% acute jump declines")
    a, b, c = st.columns(3)
    drop_options = {k: v for k, v in JUMP_METRICS.items() if v in ("ci_ns", "ci_100ms_ns", "mrsi") and v in available.values()}
    drop_options["Any of CI / CI100 / mRSI"] = "Any of CI / CI100 / mRSI"
    with a:
        drop_choice = st.selectbox("Drop metric", list(drop_options))
    with b:
        baseline = st.selectbox("Reference", ["Previous test", "Mean of previous 3 tests"])
    with c:
        acute_load_label = st.selectbox("Game workload", list(LOAD_LABELS), key="acute_load")
    acute_load = LOAD_LABELS[acute_load_label]
    try:
        summary, events = analyze_drops(snapshot, acute_load, drop_options[drop_choice],
                                        "prior1" if baseline == "Previous test" else "prior3")
        q1, q2, q3 = st.columns(3)
        q1.metric("Eligible tests with baseline", f"{len(events):,}")
        q2.metric("Drop events", f"{int(events['drop_flag'].sum()):,}")
        q3.metric("Athletes with drops", f"{events.loc[events['drop_flag'], 'Athlete'].nunique():,}")
        st.dataframe(summary.style.format(precision=2), hide_index=True, use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=summary["Window (days)"].astype(str),
                             y=summary["Within-athlete deviation before drop"],
                             name="Drop events"))
        fig.add_trace(go.Bar(x=summary["Window (days)"].astype(str),
                             y=summary["Within-athlete deviation before other tests"],
                             name="Other tests"))
        fig.update_layout(barmode="group", xaxis_title="Previous calendar days",
                          yaxis_title="Mean workload deviation from player's mean", height=390)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Descriptive, unadjusted event comparison; this does NOT estimate drop odds, "
                   "establish causality, or validate 5% as a reliable-change threshold. "
                   "Events are defined from complete pre-eligibility jump history.")
        st.dataframe(events.loc[events["drop_flag"]].sort_values("jump_date", ascending=False),
                     use_container_width=True, hide_index=True)
        st.download_button("Download all eligible event observations (CSV)",
                           events.to_csv(index=False).encode(),
                           file_name="acute_drop_events_2026.csv", mime="text/csv")
    except ValueError as exc:
        st.warning(str(exc))

with main3:
    st.subheader("Source matching and eligibility")
    c1, c2, c3 = st.columns(3)
    c1.metric("Athletes", f"{audit['eligible_athletes']:,}")
    c2.metric("Eligible athlete-days", f"{audit['eligible_athlete_days']:,}")
    c3.metric("Excluded athlete-days", f"{audit['excluded_player_days']:,}")
    st.json(audit)
    st.markdown("**Important methods notes**")
    st.markdown("- Matches `PP_Sprint.batter` to `Jump Data.Athlete` by normalized name. Check unmatched counts; provide a mapping CSV if necessary. This is not a verified MLBAM ID linkage.\n"
                "- Combines doubleheaders by player and date, removing exact duplicate game rows.\n"
                "- Excludes same-day game loads; requires 28 days after the first observed game for each athlete, and stops no later than the day after their last recorded game.\n"
                "- Missing game-date rows **inside** coverage are interpreted as zero *recorded* game load. They may also represent gaps in the source.\n"
                "- Averaging replicate trials on a date produces one jump observation per athlete-day.\n"
                "- Random-intercept-and-slope models separate within-athlete and between-athlete workload. Month and optionally test gap are adjusted.\n"
                "- Random slopes and serially correlated errors are different issues: report the lag-1 residual correlation; this app does **not** fit AR(1) residual errors.\n"
                "- The six cumulative windows overlap and are fitted separately, not as independent exposures in one model.\n"
                "- Clinical/practical significance is not established by p-values or by the descriptive 5% threshold alone.")
