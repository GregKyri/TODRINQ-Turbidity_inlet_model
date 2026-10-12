"""
Bootstrap / subagging ensemble uncertainty quantification for RF and XGBoost.

Builds on Shukla et al. (2026, ACS ES&T Water), who draw B bootstrap resamples
of the training set, fit one model per resample, and use the spread among
members as the forecast distribution, sweeping B in {2,3,5,7,10} and evaluating
with CRPS. That protocol is reproduced here and then extended in five ways,
each motivated by something the original leaves untested.

1. RESAMPLING FRACTION.  A classic bootstrap (n draws with replacement) retains
   only ~63.2% of unique rows, so every member is trained on substantially less
   information than the full dataset. SAMPLE_FRAC with REPLACE=False performs
   subagging instead (Buhlmann & Yu, 2002): each member sees exactly that
   fraction of unique rows. Raising it improves member accuracy and narrows
   member spread; the sweep quantifies that trade-off.

2. EMPIRICAL COVERAGE.  CRPS rewards a distribution that is sharp and well
   placed but does not reveal under-dispersion: an ensemble can improve CRPS
   by sharpening its mean while its 90% interval contains far fewer than 90% of
   observations. Coverage and reliability are reported throughout.

3. A PROCESS-ERROR TERM.  Resampling the training data captures PARAMETER
   uncertainty only. Adding residuals drawn from each member's out-of-bag rows
   turns the ensemble into a full predictive distribution. Residuals are
   sampled in MODEL space (log1p), so after back-transformation the intervals
   widen multiplicatively with turbidity.

4. DETERMINISTIC REFERENCES.  Each family is also fitted once on ALL training
   rows. Its CRPS equals its MAE (CRPS of a point forecast reduces to absolute
   error), so the comparison is structurally unfavourable to the deterministic
   model and must be read alongside RMSE, which the ensemble typically loses on
   because its members see less data.

   The same single models also provide the cheapest route to calibrated
   intervals: for Random Forest, out-of-bag predictions give honest residuals
   from a full-data model at no cost, so SINGLE+R is a one-model alternative to
   the whole ensemble. If it matches the ensemble's coverage at better RMSE,
   the ensemble is unnecessary and the budget table explains why.

5. CONDITIONAL DIAGNOSTICS.  Aggregate coverage can hide failure where it
   matters. Coverage is therefore also reported by quintile of observed
   turbidity, together with a PIT histogram.

CRPS is reported with both the fair (unbiased, Ferro 2014) and the biased
1/m^2 estimators. The fair estimator targets the CRPS of the underlying
distribution and is independent of ensemble size by construction, so it is flat
in B; the biased one decreases with B and is what reproduces the published
figure. Showing both separates a property of the estimator from a property of
the ensemble.

USAGE (Colab -- no argparse; sys.argv carries kernel flags)
    import bootstrap_uq, importlib
    importlib.reload(bootstrap_uq)
    res = bootstrap_uq.run(horizon=6, split="random")

OUTPUT (under <save_root>/uq/)
    bootstrap_uq_<run>.csv        CRPS / RMSE / coverage / width per configuration
    reliability_<run>.csv         empirical vs nominal coverage
    coverage_by_level_<run>.csv   coverage and width by observed-turbidity quintile
    variance_budget_<run>.csv     parameter / structural / unexplained variance
    fig_uq_skill_<run>.png        CRPS and coverage against ensemble size
    fig_uq_reliability_<run>.png  nominal vs empirical coverage
    fig_uq_series_<run>.png       observations, band and deterministic models
    fig_uq_diagnostics_<run>.png  PIT histogram and coverage by level
"""

import glob
import importlib
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.simplefilter("ignore", UserWarning)

try:
    from IPython.display import Image, display
    _NB = True
except Exception:
    _NB = False


# =============================================================================
# CONFIG
# =============================================================================
TRAIN_MODULE = "new_version_training"   # your training script, no .py
SEED = 42

# --- ensemble ----------------------------------------------------------------
B_MAX = 10
B_GRID = [1, 2, 3, 5, 7, 10]
N_SUBSETS = 20              # random member subsets averaged per B < B_MAX
SAMPLE_FRAC = 0.75          # None -> classic bootstrap (n draws, with replacement)
REPLACE = False             # False -> subagging: exactly SAMPLE_FRAC unique rows

# --- process error -----------------------------------------------------------
ADD_PROCESS_ERROR = True
N_NOISE_DRAWS = 50          # residual draws per ensemble member
MAX_OOB_RESID = 20000

# --- single-model references -------------------------------------------------
ADD_SINGLE = True           # one model on ALL rows, as a deterministic reference
ADD_SINGLE_RESID = True     # ... plus residual-based intervals (no ensemble)
N_SINGLE_DRAWS = 1000
CAL_FRACTION = 0.20         # XGBoost has no OOB; hold this out for residuals

# --- evaluation --------------------------------------------------------------
NOMINAL = 0.90
RELIABILITY_LEVELS = [0.50, 0.70, 0.80, 0.90, 0.95]
N_LEVEL_BINS = 5            # quintiles of observed turbidity

# --- figures -----------------------------------------------------------------
EVENT_PANEL_HOURS = 14 * 24

# Illustration panel: predict EVERY hour in this window, training and test
# alike, and shade the withheld hours. A continuous line needs a prediction at
# every hour; under a random split the test rows alone are ~1 hour in 5 and
# cannot support one. Set to None to skip. Carries no metrics by design.
ILLUSTRATION_WINDOW = ("2022-12-15", "2023-02-15")
FIG_DPI = 200

P = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
     "grid": "#e1e0d9", "axis": "#c3c2b7",
     "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]}
HUE = {"RF": P["series"][0], "XGB": P["series"][1], "POOL": P["series"][2]}


def _rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "figure.facecolor": P["surface"], "savefig.facecolor": P["surface"],
        "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.titlecolor": P["ink"], "axes.labelcolor": P["ink2"],
        "axes.labelsize": 10, "text.color": P["ink"],
        "legend.frameon": False, "legend.fontsize": 9})


def _ax(ax, axis="y"):
    ax.set_facecolor(P["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(P["axis"]); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=P["muted"], labelsize=9, length=0)
    if axis == "both":
        ax.grid(color=P["grid"], linewidth=0.8, zorder=0)
    elif axis:
        ax.grid(axis=axis, color=P["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    return ax


def _hue(name):
    return HUE.get(name.split("+")[0].split("_")[0], P["series"][3])


def _show(path):
    print(f"  -> {path}")
    if _NB:
        display(Image(path))


# =============================================================================
# SCORING
# =============================================================================
def crps_ensemble(y, M, fair=True):
    """Empirical CRPS = E|X - y| - 0.5 E|X - X'|, O(m log m) per point.

    fair=True uses the unbiased 1/(m(m-1)) estimator (Ferro, 2014), which
    targets the CRPS of the underlying distribution and is therefore
    independent of ensemble size. fair=False uses 1/m^2, which improves with m
    and is what most ensemble-size sweeps in the literature report.
    """
    y = np.asarray(y, float)
    M = np.asarray(M, float)
    m = M.shape[0]
    if m == 1:
        return float(np.mean(np.abs(M[0] - y)))       # point forecast: CRPS = MAE
    S = np.sort(M, axis=0)
    t1 = np.abs(M - y).mean(axis=0)
    i = np.arange(1, m + 1)[:, None]
    denom = m * (m - 1) if fair else m * m
    e_xx = (2.0 / denom) * ((2 * i - m - 1) * S).sum(axis=0)
    return float(np.mean(t1 - 0.5 * e_xx))


def interval_stats(y, M, nominal=NOMINAL):
    q = [100 * (1 - nominal) / 2, 100 * (1 + nominal) / 2]
    lo, hi = np.percentile(M, q, axis=0)
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo)), lo, hi


def coverage_by_level(y, M, nominal=NOMINAL, n_bins=N_LEVEL_BINS):
    """Does the interval cover during events, or only in calm water?"""
    _, _, lo, hi = interval_stats(y, M, nominal)
    inside = (y >= lo) & (y <= hi)
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.digitize(y, edges[1:-1]), 0, n_bins - 1)
    return pd.DataFrame([{
        "bin": f"Q{i+1}", "range_FTU": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
        "n": int((b == i).sum()),
        "coverage": float(inside[b == i].mean()) if (b == i).any() else np.nan,
        "mean_width_FTU": float((hi - lo)[b == i].mean()) if (b == i).any() else np.nan}
        for i in range(n_bins)])


def pit_values(y, M):
    """Probability integral transform: fraction of members below each observation."""
    return (np.asarray(M, float) < np.asarray(y, float)).mean(axis=0)


# =============================================================================
# DATA AND HYPERPARAMETERS
# =============================================================================
def rebuild(T, split, block_days, horizon):
    """Reconstruct the exact training/test arrays the saved models used."""
    T.SPLIT_MODE, T.BLOCK_DAYS, T.HORIZONS = split, block_days, [horizon]
    W = T.INPUT_WINDOW
    df, valid = T.load_data()
    feats = list(T.FEATURE_COLUMNS)

    X_all, y_all, ts_all, spans, persist_all = T.make_windows(
        df, valid, feats, input_window=W, forecast_horizon=horizon)
    row_block = T.row_block_ids(df.index, block_days)
    idx_tr, idx_te, groups_tr = T.split_windows(X_all, y_all, ts_all, spans, row_block)

    scalers = T.fit_window_scalers(X_all[idx_tr], y_all[idx_tr], feats, W)
    X_tr, y_tr = T.apply_window_scalers(X_all[idx_tr], y_all[idx_tr], scalers, feats, W)
    X_te, _ = T.apply_window_scalers(X_all[idx_te], y_all[idx_te], scalers, feats, W)

    cols = T.tree_column_index(feats, W)
    print(f"\n  train {X_tr[:, cols].shape}   test {X_te[:, cols].shape}")
    print("  CHECK: the test count must match the n in your results table for "
          "this scenario, or the split settings do not match the saved models.")

    is_test = np.zeros(len(X_all), bool)
    is_test[idx_te] = True

    return dict(X_tr=X_tr[:, cols], X_te=X_te[:, cols], y_tr=y_tr,
                y_te_ftu=y_all[idx_te].astype(float), ts_te=ts_all[idx_te],
                persist_te=persist_all[idx_te].astype(float),
                groups_tr=groups_tr, scalers=scalers, cols=cols, T=T,
                # kept UNSCALED for the illustration panel: scaling the whole
                # record would copy it, and only one window is ever needed
                X_all_raw=X_all, y_all_raw=y_all, ts_all=ts_all,
                persist_all=persist_all, is_test=is_test)


def illustration_slice(ctx, start, end):
    """Every hour in a window — training and test alike — scaled and ready to
    predict, with a flag saying which hours were withheld from training.

    This supports a DEMONSTRATION panel, not an evaluation: a continuous line
    needs a prediction at every hour, and the reader must be able to see which
    of them the model had already seen.
    """
    T = ctx["T"]
    feats = list(T.FEATURE_COLUMNS)
    ts = pd.DatetimeIndex(ctx["ts_all"])
    sel = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    if not sel.any():
        return None
    Xw, _ = T.apply_window_scalers(ctx["X_all_raw"][sel], ctx["y_all_raw"][sel],
                                   ctx["scalers"], feats, T.INPUT_WINDOW)
    return dict(sel=sel, ts=ts[sel], X=Xw[:, ctx["cols"]],
                y=ctx["y_all_raw"][sel].astype(float),
                persist=ctx["persist_all"][sel].astype(float),
                is_test=ctx["is_test"][sel])


def load_params(T, kind, horizon, n_features, tag=None):
    """Tuned hyperparameters from the saved config.json. Matches on horizon,
    feature count and lag tag, so an f2 config can never be loaded for an f7 run."""
    folder = {"rf": "RF models", "xgb": "XGB models"}[kind]
    pat = os.path.join(T.save_root(), folder, "*", "config.json")
    hits = []
    for h in sorted(glob.glob(pat)):
        c = json.load(open(h))
        if c.get("forecasting horizon (hours)") != horizon:
            continue
        if len(c.get("feature_columns", [])) != n_features:
            continue
        if tag and tag not in os.path.basename(os.path.dirname(h)):
            continue
        hits.append((h, c))
    if not hits:
        avail = [os.path.basename(os.path.dirname(x)) for x in glob.glob(pat)]
        raise FileNotFoundError(
            f"no {kind} config for h={horizon}, {n_features} features"
            + (f", tag '{tag}'" if tag else "") + f"\n  available: {avail}")
    h, c = hits[0]
    print(f"  {kind.upper()} params <- {os.path.basename(os.path.dirname(h))}"
          f"   ({len(c['feature_columns'])} features, "
          f"{c.get('n_input_columns', '?')} columns)")
    if len(hits) > 1:
        print(f"    ({len(hits)} matches, using the first)")
    return c["best_params"]


def make_model(kind, params, seed, T, oob=False):
    if kind == "rf":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, bootstrap=True,
                                     oob_score=oob, **params)
    return XGBRegressor(objective="reg:squarederror", tree_method="hist",
                        device=T.XGB_DEVICE, random_state=seed, n_jobs=-1, **params)


# =============================================================================
# FITTING
# =============================================================================
def fit_members(kind, params, ctx, B=B_MAX, seed=SEED,
                X_extra=None, extra_out=None):
    """B models on resampled training sets.

    SAMPLE_FRAC=None reproduces the classic bootstrap (n draws with replacement,
    ~63.2% unique). Otherwise SAMPLE_FRAC*n rows are drawn, without replacement
    when REPLACE is False, giving exactly that fraction of unique rows.
    """
    T = ctx["T"]
    X_tr, y_tr, X_te = ctx["X_tr"], ctx["y_tr"], ctx["X_te"]
    n = len(X_tr)
    rng = np.random.default_rng(seed)

    scaled, resids = [], []
    for b in range(B):
        t0 = time.time()
        if SAMPLE_FRAC is None:
            idx = rng.choice(n, n, replace=True)
        else:
            idx = rng.choice(n, int(round(SAMPLE_FRAC * n)), replace=REPLACE)
        m = make_model(kind, params, seed=b, T=T)
        m.fit(X_tr[idx], y_tr[idx])
        scaled.append(m.predict(X_te))
        if X_extra is not None and extra_out is not None:
            extra_out.append(m.predict(X_extra))   # predict now, drop the model

        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob) > MAX_OOB_RESID:
            oob = rng.choice(oob, MAX_OOB_RESID, replace=False)
        resids.append(y_tr[oob] - m.predict(X_tr[oob]))
        print(f"    member {b+1}/{B}  ({len(np.unique(idx))/n:.1%} unique, "
              f"{len(oob):,} held out, {time.time()-t0:.0f}s)")

    S = np.stack(scaled)
    F = np.stack([T.to_ftu(s, ctx["scalers"]) for s in S])
    return S, F, resids


def fit_single(kind, params, ctx, seed=SEED, X_extra=None, extra_out=None):
    """One model on ALL training rows -- the deterministic reference.

    For Random Forest, out-of-bag predictions give honest residuals from the
    full-data model at no cost. XGBoost has no OOB equivalent, so a calibration
    slice is held out; that model therefore sees slightly less data, which is
    noted in the output.
    """
    T = ctx["T"]
    X_tr, y_tr, X_te = ctx["X_tr"], ctx["y_tr"], ctx["X_te"]

    if kind == "rf":
        m = make_model(kind, params, seed, T, oob=True)
        m.fit(X_tr, y_tr)
        resid = y_tr - m.oob_prediction_
        note = "full data, OOB residuals"
    else:
        fit_i, cal_i = T.holdout_indices(len(X_tr), ctx["groups_tr"],
                                         CAL_FRACTION, seed=7)
        m = make_model(kind, params, seed, T)
        m.fit(X_tr[fit_i], y_tr[fit_i])
        resid = y_tr[cal_i] - m.predict(X_tr[cal_i])
        note = f"{100*(1-CAL_FRACTION):.0f}% of data, {len(cal_i):,} calibration rows"

    s = m.predict(X_te)
    if X_extra is not None and extra_out is not None:
        extra_out.append(m.predict(X_extra))
    resid = resid[np.isfinite(resid)]
    print(f"  {kind.upper()} single model: {note}, {len(resid):,} residuals")
    return m, s, T.to_ftu(s, ctx["scalers"]), resid


def draw_with_resid(scaled_preds, resids, ctx, n_draws, seed=1):
    """Add resampled residuals in MODEL space, then back-transform.

    Because the target is log1p(FTU), an additive residual here becomes
    multiplicative in FTU: the interval widens with the level automatically.
    `scaled_preds` is (B, n) or (n,); `resids` a matching list or one array.
    """
    T, rng = ctx["T"], np.random.default_rng(seed)
    S = np.atleast_2d(scaled_preds)
    R = resids if isinstance(resids, (list, tuple)) else [resids] * len(S)
    out = []
    for s, r in zip(S, R):
        r = np.asarray(r, float)
        for _ in range(n_draws):
            out.append(T.to_ftu(s + rng.choice(r, size=len(s)), ctx["scalers"]))
    return np.stack(out)


def sweep(F, y, b_grid=B_GRID, n_subsets=N_SUBSETS, seed=3):
    """Metrics against ensemble size, averaged over random member subsets."""
    rng = np.random.default_rng(seed)
    B = F.shape[0]
    rows = []
    for b in b_grid:
        if b > B:
            continue
        combos = ([np.arange(B)] if b == B else
                  [rng.choice(B, b, replace=False) for _ in range(n_subsets)])
        cf, cb, rm, cv, wd = [], [], [], [], []
        for c in combos:
            M = F[c]
            cf.append(crps_ensemble(y, M, fair=True))
            cb.append(crps_ensemble(y, M, fair=False))
            rm.append(float(np.sqrt(np.mean((M.mean(0) - y) ** 2))))
            if b >= 2:
                cov, w, _, _ = interval_stats(y, M)
                cv.append(cov); wd.append(w)
        rows.append({"B": b, "CRPS_fair": np.mean(cf), "CRPS_biased": np.mean(cb),
                     "RMSE_mean": np.mean(rm),
                     "coverage90": np.mean(cv) if cv else np.nan,
                     "width90_FTU": np.mean(wd) if wd else np.nan})
    return pd.DataFrame(rows)


def summarise(name, y, M, B=None):
    cov, wid, _, _ = interval_stats(y, M)
    return pd.DataFrame([{
        "model": name, "B": B if B is not None else M.shape[0],
        "CRPS_fair": crps_ensemble(y, M, fair=True),
        "CRPS_biased": crps_ensemble(y, M, fair=False),
        "RMSE_mean": float(np.sqrt(np.mean((M.mean(0) - y) ** 2))),
        "coverage90": cov, "width90_FTU": wid}])


def variance_budget(fam_bands, y):
    """Parameter / structural / unexplained shares of total error variance."""
    fams = list(fam_bands)
    within = float(np.mean([fam_bands[f].var(axis=0, ddof=1) for f in fams]))
    means = np.stack([fam_bands[f].mean(axis=0) for f in fams])
    between = float(means.var(axis=0, ddof=1).mean()) if len(fams) > 1 else 0.0
    pooled = np.concatenate([fam_bands[f] for f in fams], axis=0)
    mse = float(np.mean((pooled.mean(0) - y) ** 2))
    return pd.DataFrame([{
        "MSE_total": mse,
        "var_parameter": within, "pct_parameter": 100 * within / mse,
        "var_structural": between, "pct_structural": 100 * between / mse,
        "var_unexplained": mse - within - between,
        "pct_unexplained": 100 * (mse - within - between) / mse}])


# =============================================================================
# FIGURES
# =============================================================================
def fig_skill(curves, path, run, markers=None, single_crps=None):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for name, t in curves.items():
        c = _hue(name)
        _ax(axes[0]).plot(t["B"], t["CRPS_fair"], marker="o", markersize=5, lw=2.0,
                          color=c, label=f"{name} (fair)", zorder=3)
        _ax(axes[0]).plot(t["B"], t["CRPS_biased"], marker="^", markersize=5, lw=1.6,
                          color=c, ls=(0, (5, 2)), label=f"{name} (biased)", zorder=3)
        s = t[t["B"] >= 2]
        _ax(axes[1]).plot(s["B"], s["coverage90"], marker="o", markersize=5, lw=2.0,
                          color=c, label=name, zorder=3)

    for name, row in (markers or {}).items():
        c = _hue(name)
        axes[0].plot(row["B"], row["CRPS_fair"], marker="*", markersize=15,
                     color=c, ls="none", label=name, zorder=4)
        axes[1].plot(row["B"], row["coverage90"], marker="*", markersize=15,
                     color=c, ls="none", label=name, zorder=4)

    for name, v in (single_crps or {}).items():
        axes[0].axhline(v, color=_hue(name), lw=1.1, ls=(0, (1, 2)), zorder=2)
        axes[0].text(B_GRID[-1], v, f" {name} single", va="center", ha="right",
                     fontsize=8, color=P["muted"])

    axes[0].set_xlabel("ensemble members (B)"); axes[0].set_ylabel("CRPS (FTU)")
    axes[0].set_title("Probabilistic skill", loc="left", pad=10)
    axes[1].axhline(NOMINAL, color=P["axis"], lw=1.2, ls=(0, (4, 3)), zorder=2)
    axes[1].text(B_GRID[-1], NOMINAL + 0.02, "nominal 90%", ha="right", va="bottom",
                 fontsize=9, color=P["muted"])
    axes[1].set_xlabel("ensemble members (B)")
    axes[1].set_ylabel("empirical coverage"); axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Interval calibration", loc="left", pad=10)
    for a in axes:
        a.legend(labelcolor=P["ink2"], fontsize=8)
    fig.suptitle("Ensemble skill and calibration", x=0.005, ha="left",
                 fontsize=12, fontweight="semibold", color=P["ink"])
    fig.text(0.005, 0.925, run, fontsize=9, color=P["muted"], ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); _show(path)


def fig_reliability(rel, path, run):
    """Family on hue, calibrated variant on line style -- six series, three hues."""
    fig, ax = plt.subplots(figsize=(5.9, 5.4))
    _ax(ax, axis="both")
    ax.plot([0.4, 1], [0.4, 1], color=P["axis"], lw=1.2, ls=(0, (4, 3)), zorder=2)
    for name in rel.columns:
        wide = ("+PE" in name) or ("+R" in name)
        ax.plot(rel.index, rel[name],
                marker="s" if wide else "o", markersize=5, lw=2.0,
                color=_hue(name), ls=(0, (5, 2)) if wide else "-",
                markerfacecolor=P["surface"] if wide else None,
                label=name, zorder=3)
    ax.text(0.955, 0.985, "with process error", ha="right", va="bottom",
            fontsize=9, color=P["ink2"])
    ax.text(0.955, 0.30, "parameter spread only", ha="right", va="top",
            fontsize=9, color=P["ink2"])
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_xlim(0.45, 1.0); ax.set_ylim(0, 1.02)
    ax.set_title("Interval reliability", loc="left", pad=16)
    fig.text(0.01, 0.945, run, fontsize=9, color=P["muted"], ha="left")
    ax.legend(labelcolor=P["ink2"], loc="center left", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); _show(path)


def _break_gaps(ts, v, max_gap_h=1):
    """Insert NaN wherever consecutive TEST hours are further apart than
    max_gap_h, so no line is drawn across hours the model trained on.

    Under a random split only ~20% of hours are test rows. Joining consecutive
    rows of the test array would imply an hourly forecast that was never
    produced, and would hide the training hours sitting in every gap.
    """
    ts = pd.DatetimeIndex(ts)
    v = np.asarray(v, float)
    if len(ts) < 2:
        return ts, v
    hours = ts.to_numpy().astype("datetime64[h]").astype("int64")
    cut = np.where(np.diff(hours) > max_gap_h)[0]
    if not len(cut):
        return ts, v
    ins = (ts[cut] + pd.Timedelta(hours=1)).to_numpy()
    all_ts = np.concatenate([ts.to_numpy(), ins])
    all_v = np.concatenate([v, np.full(len(cut), np.nan)])
    o = np.argsort(all_ts, kind="stable")
    return pd.DatetimeIndex(all_ts[o]), all_v[o]


def fig_series(ts, y, M, path, run, label, singles=None, persist=None,
               window=None, hours=EVENT_PANEL_HOURS):
    """Observations, the predictive band and the deterministic models.

    The window is chosen in WALL-CLOCK time, not in rows of the test array, and
    every line is broken at gaps. Slicing k rows of a 20% random test set spans
    about 5k hours, so the previous row-based panel covered ~70 days while
    claiming 14, and drew the forecast straight through the training hours.
    """
    ts = pd.DatetimeIndex(ts)
    y = np.asarray(y, float)
    if window is not None:
        sel = (ts >= pd.Timestamp(window[0])) & (ts <= pd.Timestamp(window[1]))
    else:
        c = int(np.argmax(y))
        half = pd.Timedelta(hours=hours // 2)
        sel = (ts >= ts[c] - half) & (ts <= ts[c] + half)
    if not sel.any():
        print("  series panel skipped: no test hours in the window")
        return

    t = ts[sel]
    q = [100 * (1 - NOMINAL) / 2, 100 * (1 + NOMINAL) / 2]
    lo, hi = np.percentile(M[:, sel], q, axis=0)
    med = np.median(M[:, sel], axis=0)
    span = (t[-1] - t[0]) / pd.Timedelta("1h") + 1

    tg, log_ = _break_gaps(t, lo)
    _, hig = _break_gaps(t, hi)
    fig, ax = plt.subplots(figsize=(10.8, 4.4)); _ax(ax)
    ax.fill_between(tg, log_, hig, color=P["series"][0], alpha=0.18, lw=0,
                    label=f"{NOMINAL:.0%} interval ({label})", zorder=2)
    ax.plot(*_break_gaps(t, med), color=P["series"][0], lw=2.0,
            label=f"{label} median", zorder=5)
    for nm, p in (singles or {}).items():
        ax.plot(*_break_gaps(t, np.asarray(p)[sel]), lw=1.6, ls=(0, (5, 2)),
                color=_hue(nm), label=f"{nm} (single)", zorder=4)
    if persist is not None:
        ax.plot(*_break_gaps(t, np.asarray(persist)[sel]), lw=1.2, ls=(0, (1, 2)),
                color=P["muted"], label="persistence", zorder=3)
    ax.plot(*_break_gaps(t, y[sel]), color=P["ink"], lw=1.6,
            label="observed", zorder=6)

    ax.set_ylabel("turbidity (FTU)")
    ax.set_title("Forecast, uncertainty and deterministic models over an event",
                 loc="left", pad=22)
    ax.text(0, 1.025,
            f"{run}   ·   {t[0]:%d %b %Y} – {t[-1]:%d %b %Y}   ·   {len(t):,} "
            f"test hours of {span:,.0f} ({len(t) / span:.0%}); "
            "gaps are hours held in training",
            transform=ax.transAxes, fontsize=8.5, color=P["ink2"])
    ax.legend(labelcolor=P["ink2"], ncol=3, loc="upper left", fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); _show(path)


def fig_illustration(w, band, path, run, label, singles=None):
    """DEMONSTRATION panel: the model run continuously over a window, with the
    hours withheld from training shaded.

    Every hour has a prediction, so the line is genuinely continuous and
    nothing is hidden; in exchange the panel is not an evaluation and carries
    no metrics. Quantitative performance belongs in the table, on test rows.
    """
    t, y, is_te = w["ts"], w["y"], w["is_test"]
    q = [100 * (1 - NOMINAL) / 2, 100 * (1 + NOMINAL) / 2]
    lo, hi = np.percentile(band, q, axis=0)
    med = np.median(band, axis=0)

    fig, ax = plt.subplots(figsize=(11.8, 4.6)); _ax(ax)

    # shade the withheld hours first, so everything else sits on top
    starts, lens = [], []
    d = np.diff(np.concatenate([[0], is_te.view(np.int8), [0]]))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        starts.append(s); lens.append(e - s)
    for i, (s, L) in enumerate(zip(starts, lens)):
        ax.axvspan(t[s], t[min(s + L, len(t) - 1)], color=P["series"][3],
                   alpha=0.16, lw=0, zorder=1,
                   label="withheld from training" if i == 0 else None)

    ax.fill_between(t, lo, hi, color=P["series"][0], alpha=0.20, lw=0,
                    label=f"{NOMINAL:.0%} interval ({label})", zorder=2)
    ax.plot(t, med, color=P["series"][0], lw=2.0, label=f"{label} median",
            zorder=5)
    for nm, p in (singles or {}).items():
        ax.plot(t, np.asarray(p), lw=1.5, ls=(0, (5, 2)), color=_hue(nm),
                label=f"{nm} (single)", zorder=4)
    ax.plot(t, y, color=P["ink"], lw=1.6, label="observed", zorder=6)

    # rug strip: the shaded spans collapse to unreadable slivers when the
    # withheld hours are scattered singletons, which is exactly what a random
    # split produces. The rug shows the pattern either way.
    top = float(max(np.nanmax(hi), np.nanmax(y)))
    y0 = -0.055 * top
    ax.plot(t[is_te], np.full(int(is_te.sum()), y0), "|",
            color=P["series"][3], markersize=7, markeredgewidth=1.0, zorder=7)
    ax.set_ylim(y0 * 2.1, top * 1.30)          # headroom so the legend clears

    pct = 100 * is_te.mean()
    ax.set_ylabel("turbidity (FTU)")
    ax.set_title("Model behaviour over a selected period — illustration, "
                 "not an evaluation", loc="left", pad=22)
    ax.text(0, 1.025,
            f"{run}   ·   {t[0]:%d %b %Y} – {t[-1]:%d %b %Y}   ·   "
            f"all {len(t):,} hours predicted; shaded = the {int(is_te.sum()):,} "
            f"({pct:.0f}%) withheld from training. "
            "Metrics are reported on test hours only, in Table X.",
            transform=ax.transAxes, fontsize=8.5, color=P["ink2"])
    ax.legend(labelcolor=P["ink2"], ncol=3, loc="upper left", fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); _show(path)


def fig_diagnostics(y, bands, path, run):
    """PIT histogram for the headline band, and coverage by observed quintile."""
    head = next(iter(bands))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    pit = pit_values(y, bands[head])
    _ax(axes[0]).hist(pit, bins=20, range=(0, 1), color=P["series"][0],
                      edgecolor=P["surface"], linewidth=1.2, zorder=3)
    axes[0].axhline(len(pit) / 20, color=P["axis"], lw=1.2, ls=(0, (4, 3)), zorder=4)
    axes[0].set_xlabel("PIT value"); axes[0].set_ylabel("count")
    axes[0].set_title(f"PIT histogram — {head}", loc="left", pad=10)
    axes[0].text(0.5, 0.96, "flat = calibrated · U = too narrow · hump = too wide",
                 transform=axes[0].transAxes, ha="center", va="top",
                 fontsize=8, color=P["muted"])

    ax = _ax(axes[1])
    names = list(bands)[:3]
    w = 0.8 / len(names)
    for i, nm in enumerate(names):
        cb = coverage_by_level(y, bands[nm])
        x = np.arange(len(cb)) + (i - (len(names) - 1) / 2) * w
        ax.bar(x, cb["coverage"], width=w * 0.92, color=_hue(nm), label=nm,
               edgecolor=P["surface"], linewidth=1.0, zorder=3)
    ax.axhline(NOMINAL, color=P["axis"], lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.set_xticks(np.arange(len(cb)))
    ax.set_xticklabels([f"Q{i+1}\n{r}" for i, r in enumerate(cb["range_FTU"])],
                       fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("empirical coverage")
    ax.set_xlabel("quintile of observed turbidity (FTU)")
    ax.set_title("Coverage by observed level", loc="left", pad=10)
    ax.legend(labelcolor=P["ink2"], fontsize=8, loc="lower left")

    fig.suptitle("Distributional diagnostics", x=0.005, ha="left", fontsize=12,
                 fontweight="semibold", color=P["ink"])
    fig.text(0.005, 0.925, run, fontsize=9, color=P["muted"], ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); _show(path)


# =============================================================================
# MAIN
# =============================================================================
def run(horizon=6, split="random", block_days=30, models=("rf", "xgb"),
        train_module=TRAIN_MODULE, add_process_error=ADD_PROCESS_ERROR,
        add_single=ADD_SINGLE, add_single_resid=ADD_SINGLE_RESID,
        pool_families=True):
    _rc()
    T = importlib.import_module(train_module)
    importlib.reload(T)
    T.SPLIT_MODE, T.BLOCK_DAYS = split, block_days

    nf = len(T.FEATURE_COLUMNS)
    frac = "bootstrap (63.2% unique)" if SAMPLE_FRAC is None else \
           f"{SAMPLE_FRAC:.0%} {'with' if REPLACE else 'without'} replacement"
    print(f"FEATURE_COLUMNS ({nf}): {list(T.FEATURE_COLUMNS)}")
    print(f"tree lags: {T.tree_tag()}   horizon: {horizon}   split: {split}")
    print(f"resampling: {frac}")

    ctx = rebuild(T, split, block_days, horizon)
    y = ctx["y_te_ftu"]
    out_dir = os.path.join(T.save_root(), "uq"); os.makedirs(out_dir, exist_ok=True)
    run_name = f"{T.split_tag()}_{T.tree_tag()}_h{horizon}_f{nf}"
    if SAMPLE_FRAC is not None:
        run_name += f"_s{int(round(100*SAMPLE_FRAC))}"

    tables, bands, fam_bands, curves = {}, {}, {}, {}
    singles_ftu, single_crps, markers = {}, {}, {}

    # ---- illustration window: every hour, training and test alike ----------
    w = illus = None
    illus_singles, illus_fam = {}, {}
    if ILLUSTRATION_WINDOW:
        w = illustration_slice(ctx, *ILLUSTRATION_WINDOW)
        if w is None:
            print(f"\nillustration window {ILLUSTRATION_WINDOW} is outside the "
                  "record; panel skipped")
        else:
            print(f"\nillustration window: {len(w['ts']):,} hours, "
                  f"{int(w['is_test'].sum()):,} ({100 * w['is_test'].mean():.0f}%) "
                  "withheld from training")

    for kind in models:
        label = kind.upper()
        params = load_params(T, kind, horizon, nf, T.tree_tag())

        # ---- deterministic reference on all training rows ----
        if add_single:
            print(f"\n--- {label}: single model ---")
            ex = [] if w is not None else None
            _, s_single, p_single, s_resid = fit_single(
                kind, params, ctx, X_extra=(w["X"] if w is not None else None),
                extra_out=ex)
            if ex:
                illus_singles[label] = T.to_ftu(ex[0], ctx["scalers"])
            singles_ftu[label] = p_single
            mae = float(np.mean(np.abs(p_single - y)))
            single_crps[label] = mae
            tables[f"{label}_single"] = pd.DataFrame([{
                "model": f"{label} single", "B": 1,
                "CRPS_fair": mae, "CRPS_biased": mae,       # point forecast: CRPS = MAE
                "RMSE_mean": float(np.sqrt(np.mean((p_single - y) ** 2))),
                "coverage90": np.nan, "width90_FTU": np.nan}])

            if add_single_resid:
                D = draw_with_resid(s_single, s_resid, ctx, N_SINGLE_DRAWS)
                tables[f"{label}_single+R"] = summarise(f"{label} single+R", y, D)
                bands[f"{label}+R"] = D

        # ---- resampled ensemble ----
        print(f"\n--- {label}: {B_MAX} members ---")
        ex = [] if w is not None else None
        S, F, resids = fit_members(
            kind, params, ctx, B=B_MAX,
            X_extra=(w["X"] if w is not None else None), extra_out=ex)
        if ex:
            illus_fam[label] = draw_with_resid(
                np.stack(ex), resids, ctx, N_NOISE_DRAWS)
        t = sweep(F, y); t.insert(0, "model", label)
        curves[label] = t
        tables[label] = t
        bands[label] = F
        fam_bands[label] = F

        if add_process_error:
            print(f"  + process error: {N_NOISE_DRAWS} residual draws per member")
            A = draw_with_resid(S, resids, ctx, N_NOISE_DRAWS)
            tables[f"{label}+PE"] = summarise(f"{label}+PE", y, A, B=B_MAX)
            bands[f"{label}+PE"] = A
            markers[f"{label}+PE"] = {"B": B_MAX,
                                      "CRPS_fair": crps_ensemble(y, A),
                                      "coverage90": interval_stats(y, A)[0]}

    # ---- pooled across families: adds STRUCTURAL uncertainty ----
    if pool_families and len(fam_bands) > 1:
        POOL = np.concatenate(list(fam_bands.values()), axis=0)
        tables["POOL"] = summarise("POOL(" + "+".join(fam_bands) + ")", y, POOL)
        bands["POOL"] = POOL
        pe = [k for k in bands if k.endswith("+PE")]
        if pe:
            PP = np.concatenate([bands[k] for k in pe], axis=0)
            tables["POOL+PE"] = summarise("POOL+PE", y, PP)
            bands["POOL+PE"] = PP

        budget = variance_budget(fam_bands, y)
        budget.to_csv(os.path.join(out_dir, f"variance_budget_{run_name}.csv"),
                      index=False)
        print("\nUNCERTAINTY BUDGET (FTU^2)")
        print(budget.round(3).to_string(index=False))

    # ---- tables ----
    res = pd.concat(tables.values(), ignore_index=True)
    res.to_csv(os.path.join(out_dir, f"bootstrap_uq_{run_name}.csv"), index=False)

    rel = pd.DataFrame(
        {k: [interval_stats(y, M, nominal=q)[0] for q in RELIABILITY_LEVELS]
         for k, M in bands.items()}, index=RELIABILITY_LEVELS)
    rel.index.name = "nominal"
    rel.to_csv(os.path.join(out_dir, f"reliability_{run_name}.csv"))

    lev = pd.concat([coverage_by_level(y, M).assign(model=k) for k, M in bands.items()],
                    ignore_index=True)
    lev = lev[["model", "bin", "range_FTU", "n", "coverage", "mean_width_FTU"]]
    lev.to_csv(os.path.join(out_dir, f"coverage_by_level_{run_name}.csv"), index=False)

    print("\n" + "=" * 86)
    print(f"ENSEMBLE UQ   [{run_name}]")
    print("=" * 86)
    print(res.round(4).to_string(index=False))
    print("\nNote: CRPS of a deterministic forecast equals its MAE, so the single-model "
          "rows\n      are structurally penalised for having no spread. Read them with "
          "RMSE_mean.")
    print("\nRELIABILITY")
    print(rel.round(3).to_string())
    print("\nCOVERAGE BY OBSERVED LEVEL (90% interval)")
    print(lev.round(3).to_string(index=False))

    # ---- figures ----
    fig_skill(curves, os.path.join(out_dir, f"fig_uq_skill_{run_name}.png"),
              run_name, markers=markers, single_crps=single_crps)
    fig_reliability(rel, os.path.join(out_dir, f"fig_uq_reliability_{run_name}.png"),
                    run_name)

    order = [k for k in ("POOL+PE", "XGB+PE", "RF+PE", "XGB+R", "RF+R") if k in bands]
    head = order[0] if order else max(bands, key=lambda k: bands[k].shape[0])
    fig_series(ctx["ts_te"], y, bands[head],
               os.path.join(out_dir, f"fig_uq_series_{run_name}.png"),
               run_name, head, singles=singles_ftu, persist=ctx["persist_te"])

    if w is not None and illus_fam:
        illus = (np.concatenate(list(illus_fam.values()), axis=0)
                 if len(illus_fam) > 1 else list(illus_fam.values())[0])
        ilabel = "POOL+PE" if len(illus_fam) > 1 else f"{list(illus_fam)[0]}+PE"
        fig_illustration(w, illus,
                         os.path.join(out_dir, f"fig_uq_illustration_{run_name}.png"),
                         run_name, ilabel, singles=illus_singles)
    fig_diagnostics(y, {k: bands[k] for k in (order[:3] or list(bands)[:3])},
                    os.path.join(out_dir, f"fig_uq_diagnostics_{run_name}.png"),
                    run_name)

    print("\nDone.")
    return {"table": res, "reliability": rel, "by_level": lev,
            "bands": bands, "singles": singles_ftu, "ctx": ctx}