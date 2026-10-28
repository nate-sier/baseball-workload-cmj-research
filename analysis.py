"""Random-intercept + random-workload-slope ML mixed models; no silent OLS fallback."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _within_sd(data: pd.DataFrame, field: str) -> float:
    deviations = data[field] - data.groupby("player_key")[field].transform("mean")
    df = len(data) - data["player_key"].nunique()
    return float(np.sqrt(np.square(deviations).sum() / df)) if df > 0 else float("nan")


def _model_data(source: pd.DataFrame, x_col: str, y_col: str, adjust_gap: bool = True) -> tuple[pd.DataFrame, dict]:
    columns = ["player_key", "jump_date", x_col, y_col, "days_since_prior_jump"]
    d = source[columns].copy()
    d["jump_date"] = pd.to_datetime(d["jump_date"], errors="coerce")
    for key in [x_col, y_col, "days_since_prior_jump"]:
        d[key] = pd.to_numeric(d[key], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["player_key", "jump_date", x_col, y_col])
    if adjust_gap:
        d = d.dropna(subset=["days_since_prior_jump"])
    d = d.reset_index(drop=True)
    if len(d) < 50 or d["player_key"].nunique() < 5:
        raise ValueError("Need at least 50 observations across at least 5 athletes")
    x_sd = _within_sd(d, x_col)
    y_sd = _within_sd(d, y_col)
    if not np.isfinite(x_sd) or x_sd < 1e-9:
        raise ValueError("No meaningful within-athlete workload variation in this window")
    if not np.isfinite(y_sd) or y_sd < 1e-9:
        raise ValueError("No meaningful within-athlete outcome variation")
    d["x_orig"] = d[x_col]
    d["y"] = d[y_col]
    athlete_mean_x = d.groupby("player_key")["x_orig"].transform("mean")
    d["x_within"] = (d["x_orig"] - athlete_mean_x) / x_sd
    d["x_between"] = (athlete_mean_x - d["x_orig"].mean()) / x_sd
    d["month"] = d["jump_date"].dt.month.astype(str)
    d["gap"] = d["days_since_prior_jump"].clip(lower=1, upper=30)
    by_athlete_x = d.groupby("player_key")["x_orig"].std().dropna()
    by_athlete_y = d.groupby("player_key")["y"].std().dropna()
    stats = {"x_sd_within": x_sd, "y_sd_within": y_sd,
             "x_sd_median": float(by_athlete_x.median()), "y_sd_median": float(by_athlete_y.median()),
             "x_sd_overall": float(d["x_orig"].std(ddof=1)), "mean_y": float(d["y"].mean()),
             "n": len(d), "athletes": int(d["player_key"].nunique())}
    return d, stats


def _fit(d: pd.DataFrame, random_slope: bool, adjust_gap: bool):
    formula = "y ~ x_within + x_between"
    if d["month"].nunique() > 1:
        formula += " + C(month)"
    if adjust_gap and d["gap"].nunique() > 1:
        formula += " + gap"
    model = smf.mixedlm(formula, data=d, groups=d["player_key"],
                        re_formula="1 + x_within" if random_slope else "1")
    attempts = []
    for method in ("lbfgs", "bfgs"):
        try:
            with warnings.catch_warnings(record=True) as capture:
                warnings.simplefilter("always")
                result = model.fit(reml=False, method=method, maxiter=250, disp=False)
            messages = [str(w.message) for w in capture]
            if bool(result.converged):
                return result, messages
            attempts.append(f"{method}: nonconvergence; " + "; ".join(messages[:2]))
        except (ValueError, np.linalg.LinAlgError, IndexError, ZeroDivisionError) as exc:
            attempts.append(f"{method}: {type(exc).__name__}: {str(exc)[:120]}")
    raise RuntimeError("Random-" + ("slope" if random_slope else "intercept") +
                       " model did not converge: " + " | ".join(attempts))


def fit_window(source: pd.DataFrame, load: str, metric: str, days: int, *, adjust_gap: bool = True,
               compare_intercept: bool = False) -> dict:
    x_col = f"{load}_{days}d"
    d, stats = _model_data(source, x_col, metric, adjust_gap=adjust_gap)
    result, warnings_list = _fit(d, True, adjust_gap)
    beta = float(result.fe_params["x_within"])
    ci = result.conf_int().loc["x_within"]
    lower, upper = float(ci.iloc[0]), float(ci.iloc[1])
    # Random-slope variance in jump units per pooled within-athlete workload SD.
    random_sd = float(np.sqrt(max(0.0, result.cov_re.loc["x_within", "x_within"])))
    # Variance partition includes contribution of random slopes at observed within-X values.
    fixed_pred = np.asarray(result.predict(d), dtype=float)
    v_fixed = float(np.var(fixed_pred, ddof=0))
    z = np.column_stack((np.ones(len(d)), d["x_within"].to_numpy(dtype=float)))
    cov = np.asarray(result.cov_re, dtype=float)
    v_random = float(np.mean(np.einsum("ij,jk,ik->i", z, cov, z)))
    v_error = float(result.scale)
    denom = v_fixed + v_random + v_error
    # Conditional residuals remove the estimated individual intercept & slope.
    residual = np.asarray(result.resid, dtype=float)
    paired = d[["player_key", "jump_date"]].copy()
    paired["residual"] = residual
    paired = paired.sort_values(["player_key", "jump_date"])
    paired["previous_resid"] = paired.groupby("player_key")["residual"].shift()
    paired["gap_days"] = (paired["jump_date"] - paired.groupby("player_key")["jump_date"].shift()).dt.days
    pairs = paired.dropna(subset=["previous_resid", "gap_days"])
    if len(pairs) >= 5 and pairs["residual"].std() > 0 and pairs["previous_resid"].std() > 0:
        autocorr = float(pairs[["residual", "previous_resid"]].corr().iloc[0, 1])
    else:
        autocorr = np.nan
    row = {
        "Window (days)": days, "Athletes": stats["athletes"], "Jump observations": stats["n"],
        "Effect (jump units / within-X SD)": beta,
        "95% CI lower": lower, "95% CI upper": upper,
        "p-value": float(result.pvalues["x_within"]),
        "Within-X SD": stats["x_sd_within"], "Median athlete X SD": stats["x_sd_median"],
        "Within-Y SD": stats["y_sd_within"], "Median athlete Y SD": stats["y_sd_median"],
        "Standardized effect": beta / stats["y_sd_within"],
        "Standardized CI lower": lower / stats["y_sd_within"],
        "Standardized CI upper": upper / stats["y_sd_within"],
        "Effect (% mean jump)": 100 * beta / stats["mean_y"] if stats["mean_y"] else np.nan,
        "Random-slope SD": random_sd,
        "Marginal R²": v_fixed / denom if denom else np.nan,
        "Conditional R²": (v_fixed + v_random) / denom if denom else np.nan,
        "Random-slope AIC": float(result.aic), "Converged": bool(result.converged),
        "Lag-1 residual r": autocorr, "Residual pairs": int(len(pairs)),
        "Median residual pair gap (days)": float(pairs["gap_days"].median()) if len(pairs) else np.nan,
        "Warnings": "; ".join(warnings_list)[:550],
    }
    if compare_intercept:
        try:
            intercept_fit, _ = _fit(d, False, adjust_gap)
            row["Random-intercept AIC"] = float(intercept_fit.aic)
            row["AIC difference (slope - intercept)"] = float(result.aic - intercept_fit.aic)
        except RuntimeError as exc:
            row["Random-intercept AIC"] = np.nan
            row["AIC difference (slope - intercept)"] = np.nan
            row["Warnings"] += "; intercept comparison failed: " + str(exc)[:160]
    return row


def analyze_drops(df: pd.DataFrame, load: str, metric: str, reference: str, threshold: float = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive event comparison. NOT a mixed logistic regression."""
    if metric == "Any of CI / CI100 / mRSI":
        columns = [f"drop_pct_{reference}_{m}" for m in ("ci_ns", "ci_100ms_ns", "mrsi") if f"drop_pct_{reference}_{m}" in df]
    else:
        columns = [f"drop_pct_{reference}_{metric}"]
    if not columns or any(c not in df for c in columns):
        raise ValueError("Selected drop metric is unavailable in Jump Data")
    d = df.copy()
    eligible = d[columns].notna().any(axis=1)
    d = d.loc[eligible].copy()
    d["drop_flag"] = (d[columns] <= -abs(threshold)).any(axis=1)
    d["largest_pct_drop"] = d[columns].min(axis=1)
    rows = []
    for days in (1, 3, 7, 14, 21, 28):
        colx = f"{load}_{days}d"
        # Athletes differ in typical workload; compute descriptive within-player deviation.
        xw = d[colx] - d.groupby("player_key")[colx].transform("mean")
        dd = d.assign(within_workload=xw)
        event = dd.loc[dd["drop_flag"]]
        other = dd.loc[~dd["drop_flag"]]
        rows.append({"Window (days)": days,
                     "Drop events": len(event), "Non-drop tests": len(other),
                     "Mean workload before drop": event[colx].mean(),
                     "Mean workload before other tests": other[colx].mean(),
                     "Within-athlete deviation before drop": event["within_workload"].mean(),
                     "Within-athlete deviation before other tests": other["within_workload"].mean()})
    export_cols = ["Athlete", "jump_date", "largest_pct_drop", "drop_flag", *columns,
                   *[f"{load}_{w}d" for w in (1, 3, 7, 14, 21, 28)]]
    return pd.DataFrame(rows), d[export_cols].copy()
