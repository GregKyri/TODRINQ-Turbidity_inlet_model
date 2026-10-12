"""
Turbidity EVENT CLASSIFICATION at h = 3 and h = 6, RF and XGBoost.

An hour is an event if observed turbidity exceeds the threshold AND it belongs
to a run of at least MIN_RUN consecutive hours above it. The run filter is what
separates a turbidity episode from a single-hour sensor spike.

    threshold = PERCENTILE-th percentile of the TRAINING target values only.

Reuses the data pipeline of the regression script (load_data, make_windows,
block_split_indices, scalers, tree_column_index), so features, lags and the
window definition are identical to the regression chapter.

-----------------------------------------------------------------------------
IMBALANCE
-----------------------------------------------------------------------------
The training ratio is ~0.11 (1:8) with ~3,100 positive hours -- mild, and far
outside the regime synthetic oversampling targets. Three handlings are run so
the comparison can be reported rather than assumed:

    "none"    natural class distribution
    "weight"  class_weight="balanced_subsample" (RF) / scale_pos_weight (XGB)
    "adasyn"  ADASYN oversampling of the training set  (needs imbalanced-learn)

In every case the DECISION THRESHOLD is tuned on a held-out set of training
blocks, never on the test set. Most of what resampling appears to fix at this
ratio is really a threshold that was left at 0.5.

METRICS
    F1, Recall, FDR (= FP/(FP+TP) = 1 - precision) as requested, plus
    precision, PR-AUC, the confusion matrix, the hydrological triplet
    (POD / FAR / CSI) and EVENT-LEVEL detection: the fraction of test episodes
    flagged at least once within EVENT_TOLERANCE_H hours of their onset.

"""

import glob
import importlib
import inspect
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier, XGBRegressor

try:
    from imblearn.over_sampling import ADASYN
    _HAS_ADASYN = True
except ImportError:
    _HAS_ADASYN = False

try:
    from IPython.display import Image, display
    _NB = True
except Exception:
    _NB = False


# =============================================================================
# CONFIG
# =============================================================================
TRAIN_MODULE = "new_version_training"
HORIZONS = (3, 6)
BLOCK_DAYS = 30
SEED = 42

# --- event definition --------------------------------------------------------
PERCENTILE = 90.0           # of TRAINING targets
MIN_RUN = 3                 # consecutive hours above threshold
ROUND_THRESHOLD = None      # e.g. 23.5 to report a rounded value; None = exact

# --- imbalance handling ------------------------------------------------------
MODES = ("none", "weight", "adasyn")

# --- decision threshold ------------------------------------------------------
TUNE_THRESHOLD = True
VAL_FRACTION = 0.20         # training blocks held out to pick the threshold
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 3)

# --- evaluation --------------------------------------------------------------
EVENT_TOLERANCE_H = 2       # a hit counts within +/- this many hours of onset

# --- companion regression model ----------------------------------------------
FIT_REGRESSOR = True        # overlaid on the confusion plot and scored as a
                            # second baseline ("event" = predicted FTU > threshold)

# Load the ALREADY-TRAINED regression model per horizon instead of refitting.
#   {horizon: family} -- set to {} to refit on the block split instead.
# WARNING: those models were trained on the RANDOM split, so most hours in the
# block-split TEST set were in their TRAINING set. The overlaid line is then
# largely in-sample and flatters the model. The overlap is measured and printed
# at run time, and the affected rows are labelled in the output table.
SAVED_REGRESSORS = {3: "rf", 6: "xgb"}
REPORT_OVERLAP = True       # quantify the in-sample fraction (one extra reload)

# --- model persistence -------------------------------------------------------
SAVE_MODELS = True          # keep the best classifier per horizon (by F1)

# --- confusion-series figure -------------------------------------------------
PLOT_START = "2022-12-15"
PLOT_END = "2023-02-15"
MAX_GAP_H = 1               # break the line where the test set is not contiguous
CONF = {                    # colour AND marker differ, so hue is never load-bearing
    "TN": dict(c="#898781", m=".", s=9,  z=3, lab="True negative"),
    "TP": dict(c="#008300", m="o", s=26, z=5, lab="True positive"),
    "FP": dict(c="#2a78d6", m="^", s=34, z=6, lab="False positive"),
    "FN": dict(c="#e34948", m="X", s=40, z=7, lab="False negative"),
}

# --- hyperparameters ---------------------------------------------------------
# The tuned configs live under the split they were tuned on. The DATA split
# here is block; the hyperparameters are reused from wherever they exist.
PARAMS_SPLIT_TAG = "random80"
RF_DROP = {"criterion", "oob_score", "bootstrap"}
XGB_DROP = {"objective", "eval_metric", "base_score", "device", "tree_method"}
RF_FALLBACK = dict(n_estimators=500, max_depth=None, min_samples_leaf=1,
                   min_samples_split=2, max_features="sqrt")
XGB_FALLBACK = dict(n_estimators=600, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=1)

# --- figures -----------------------------------------------------------------
FIG_DPI = 200
P = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
     "grid": "#e1e0d9", "axis": "#c3c2b7",
     "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]}
HUE = {"RF": P["series"][0], "XGB": P["series"][1]}
MODE_STYLE = {"none": (0, (1, 2)), "weight": "-", "adasyn": (0, (5, 2))}


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


def _show(path):
    print(f"  -> {path}")
    if _NB:
        display(Image(path))


# =============================================================================
# EVENT DEFINITION
# =============================================================================
def run_lengths(flag):
    """Start index and length of every run of True in a boolean array."""
    flag = np.asarray(flag, bool)
    if not flag.any():
        return np.empty(0, int), np.empty(0, int)
    d = np.diff(np.concatenate([[0], flag.view(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return starts, ends - starts


def event_flag(values, threshold, min_run=MIN_RUN):
    """True where the value exceeds the threshold AND sits in a run >= min_run."""
    above = np.asarray(values, float) > threshold
    out = np.zeros_like(above)
    starts, lens = run_lengths(above)
    for s, L in zip(starts, lens):
        if L >= min_run:
            out[s:s + L] = True
    return out


def describe_events(flag, label=""):
    starts, lens = run_lengths(flag)
    if len(lens) == 0:
        print(f"  {label}: no events")
        return {}
    d = {"n_event_hours": int(flag.sum()), "n_events": int(len(lens)),
         "median_duration_h": float(np.median(lens)),
         "mean_duration_h": float(lens.mean()),
         "min_duration_h": int(lens.min()), "max_duration_h": int(lens.max()),
         "event_ratio": float(flag.mean())}
    print(f"  {label}: {d['n_event_hours']:,} event hours in {d['n_events']:,} "
          f"episodes (ratio {d['event_ratio']:.4f}); duration median "
          f"{d['median_duration_h']:.0f} h, mean {d['mean_duration_h']:.1f} h, "
          f"max {d['max_duration_h']:,} h")
    return d


# =============================================================================
# DATA
# =============================================================================
def _windows(T, horizon, block_days):
    """Windows plus the block ids, without splitting."""
    W = T.INPUT_WINDOW
    df, valid = T.load_data()
    feats = list(T.FEATURE_COLUMNS)
    X, y, ts, spans, pers = T.make_windows(
        df, valid, feats, input_window=W, forecast_horizon=horizon)
    return df, feats, X, y, ts, spans, pers, T.row_block_ids(df.index, block_days)


def random_train_overlap(T, horizon, ts_test, block_days=BLOCK_DAYS):
    """Fraction of these test hours that the RANDOM-split models trained on."""
    keep = T.SPLIT_MODE
    try:
        T.SPLIT_MODE = "random"
        _, _, X, y, ts, spans, _, rb = _windows(T, horizon, block_days)
        idx_tr, _, _ = T.split_windows(X, y, ts, spans, rb)
        return float(pd.DatetimeIndex(ts_test).isin(ts[idx_tr]).mean())
    finally:
        T.SPLIT_MODE = keep


def build(T, horizon, block_days=BLOCK_DAYS):
    """Block-split windows, scaled features, and event labels."""
    T.SPLIT_MODE, T.BLOCK_DAYS, T.HORIZONS = "block", block_days, [horizon]
    W = T.INPUT_WINDOW
    df, feats, X_all, y_all, ts_all, spans, persist_all, row_block = _windows(
        T, horizon, block_days)
    idx_tr, idx_te, groups_tr = T.split_windows(X_all, y_all, ts_all,
                                                spans, row_block)

    # --- threshold from TRAINING targets only --------------------------------
    thr_exact = float(np.percentile(y_all[idx_tr], PERCENTILE))
    thr = float(ROUND_THRESHOLD) if ROUND_THRESHOLD is not None else thr_exact
    print(f"\n  P{PERCENTILE:.0f} of training targets = {thr_exact:.2f} FTU"
          + (f"  -> using {thr:.2f} FTU" if ROUND_THRESHOLD is not None else ""))

    # --- labels on the CONTINUOUS series, then mapped to window targets ------
    # The run filter needs consecutive hours, so it is evaluated on the full
    # hourly record; only the THRESHOLD is estimated from training data.
    ev_series = pd.Series(
        event_flag(df[T.TARGET].to_numpy(float), thr, MIN_RUN), index=df.index)
    lab_all = ev_series.reindex(ts_all).to_numpy(bool)

    print("\nEVENT LABELS")
    describe_events(ev_series.to_numpy(), "full record")
    y_tr_lab, y_te_lab = lab_all[idx_tr], lab_all[idx_te]
    print(f"  training windows: {y_tr_lab.sum():,} / {len(y_tr_lab):,} "
          f"(ratio {y_tr_lab.mean():.4f})")
    print(f"  testing  windows: {y_te_lab.sum():,} / {len(y_te_lab):,} "
          f"(ratio {y_te_lab.mean():.4f})")

    scalers = T.fit_window_scalers(X_all[idx_tr], y_all[idx_tr], feats, W)
    X_tr, y_tr_reg = T.apply_window_scalers(X_all[idx_tr], y_all[idx_tr],
                                            scalers, feats, W)
    X_te, _ = T.apply_window_scalers(X_all[idx_te], y_all[idx_te], scalers, feats, W)
    cols = T.tree_column_index(feats, W)

    return dict(X_tr=X_tr[:, cols], X_te=X_te[:, cols],
                y_tr=y_tr_lab, y_te=y_te_lab,
                y_tr_reg=y_tr_reg,                      # scaled log1p target
                X_te_raw=X_all[idx_te],                 # UNSCALED: a saved model
                y_te_raw=y_all[idx_te],                 # brings its own scalers
                ftu_te=y_all[idx_te].astype(float),
                persist_te=persist_all[idx_te].astype(float),
                ts_te=ts_all[idx_te], groups_tr=groups_tr,
                scalers=scalers, cols=cols, threshold=thr, T=T)


def load_params(T, kind, horizon, n_features, tag, split_tag=PARAMS_SPLIT_TAG,
                for_classifier=True):
    """The tuned REGRESSION hyperparameters, optionally filtered for a classifier."""
    folder = {"rf": "RF models", "xgb": "XGB models"}[kind]
    pat = os.path.join(f"trained_models_{split_tag}", folder, "*", "config.json")
    for h in sorted(glob.glob(pat)):
        c = json.load(open(h))
        if (c.get("forecasting horizon (hours)") == horizon
                and len(c.get("feature_columns", [])) == n_features
                and tag in os.path.basename(os.path.dirname(h))):
            p = dict(c["best_params"])
            if for_classifier:
                drop = RF_DROP if kind == "rf" else XGB_DROP
                p = {k: v for k, v in p.items() if k not in drop}
            print(f"  {kind.upper()} params <- "
                  f"{os.path.basename(os.path.dirname(h))} (regression config)")
            return p
    print(f"  {kind.upper()}: no tuned config for h={horizon}; using defaults")
    return dict(RF_FALLBACK if kind == "rf" else XGB_FALLBACK)


def load_saved_regressor(T, kind, horizon, n_features, tag,
                         split_tag=PARAMS_SPLIT_TAG):
    """The regression model and scalers already trained by the main script.

    Layout written by save_model():
        trained_models_<split>/<RF|XGB> models/<run>/<kind>_<split>_<tag>_h<h>.joblib
        trained_models_<split>/scaler_dict_<run>.pkl
    with run = <split>_<tag>_h<horizon>_f<n_features>.
    """
    run = f"{split_tag}_{tag}_h{horizon}_f{n_features}"
    root = f"trained_models_{split_tag}"
    folder = {"rf": "RF models", "xgb": "XGB models"}[kind]
    hits = sorted(glob.glob(os.path.join(
        root, folder, run, f"{kind}_*_h{horizon}.joblib")))
    sc_path = os.path.join(root, f"scaler_dict_{run}.pkl")
    if not hits:
        print(f"    no saved {kind.upper()} model in {root}/{folder}/{run}/")
        return None
    if not os.path.exists(sc_path):
        print(f"    found the model but not {os.path.basename(sc_path)}; "
              "cannot rescale without it")
        return None
    print(f"    loaded {os.path.basename(hits[0])}")
    print(f"           {os.path.basename(sc_path)}")
    return joblib.load(hits[0]), joblib.load(sc_path), run


def predict_saved_regressor(model, scalers, ctx, T):
    """Score the block-split test windows with a model that carries its own
    scalers: rescale the RAW windows with those, not with the block scalers."""
    feats = list(T.FEATURE_COLUMNS)
    Xs, _ = T.apply_window_scalers(ctx["X_te_raw"], ctx["y_te_raw"],
                                   scalers, feats, T.INPUT_WINDOW)
    Xs = Xs[:, ctx["cols"]]
    exp = getattr(model, "n_features_in_", Xs.shape[1])
    if exp != Xs.shape[1]:
        raise ValueError(
            f"saved model expects {exp} columns, the current lag set gives "
            f"{Xs.shape[1]} -- TREE_LAG_OFFSETS does not match the saved run")
    return T.to_ftu(model.predict(Xs), scalers).astype(float)


def fit_regressor(kind, params, ctx, seed=SEED):
    """Companion regression model: same family, lags and split. Returns the
    test prediction in FTU."""
    T = ctx["T"]
    if kind == "rf":
        m = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    else:
        m = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                         device=getattr(T, "XGB_DEVICE", "cpu"),
                         random_state=seed, n_jobs=-1, **params)
    m.fit(ctx["X_tr"], ctx["y_tr_reg"])
    return T.to_ftu(m.predict(ctx["X_te"]), ctx["scalers"]).astype(float)


# =============================================================================
# MODELS
# =============================================================================
def save_bundle(path, model, ctx, kind, mode, horizon, decision_threshold,
                params, metrics):
    """Everything needed to score new data: the fitted model, the window
    scalers, the lag-column index and both thresholds. A model saved without
    the scalers and the decision threshold cannot be reused."""
    T = ctx["T"]
    b = {"model": model, "scalers": ctx["scalers"],
         "tree_columns": np.asarray(ctx["cols"], int),
         "feature_columns": list(T.FEATURE_COLUMNS),
         "input_window": int(T.INPUT_WINDOW),
         "tree_lag_offsets": sorted({int(v) for v in T.TREE_LAG_OFFSETS})
                             if T.TREE_LAG_OFFSETS else None,
         "horizon": int(horizon), "kind": kind, "imbalance_mode": mode,
         "decision_threshold": float(decision_threshold),
         "event_threshold_FTU": float(ctx["threshold"]),
         "percentile": float(PERCENTILE), "min_run": int(MIN_RUN),
         "split": "block", "block_days": int(T.BLOCK_DAYS),
         "hyperparameters": params,
         "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating))
                         else v) for k, v in metrics.items()},
         "versions": {"sklearn": sklearn.__version__,
                      "xgboost": xgboost.__version__,
                      "numpy": np.__version__},
         "created": pd.Timestamp.now().isoformat(timespec="seconds")}
    joblib.dump(b, path, compress=3)

    side = {k: v for k, v in b.items()
            if k not in ("model", "scalers", "tree_columns")}
    side["tree_columns_n"] = int(len(b["tree_columns"]))
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(side, f, indent=2, default=str)
    mb = os.path.getsize(path) / 1e6
    print(f"      saved -> {os.path.basename(path)} ({mb:.1f} MB) + .json")
    return b


def load_bundle(path):
    """Reload a saved classifier bundle."""
    return joblib.load(path)


def predict_events(bundle, df, valid, T):
    """Apply a saved bundle to new data. Returns timestamp, probability and
    the binary event call using the bundle's own decision threshold."""
    W = bundle["input_window"]
    feats = bundle["feature_columns"]
    X, y, ts, _, _ = T.make_windows(df, valid, feats, input_window=W,
                                    forecast_horizon=bundle["horizon"])
    Xs, _ = T.apply_window_scalers(X, y, bundle["scalers"], feats, W)
    p = bundle["model"].predict_proba(Xs[:, bundle["tree_columns"]])[:, 1]
    return pd.DataFrame({"timestamp": ts, "event_probability": p,
                         "event": (p >= bundle["decision_threshold"]).astype(int)})


def make_clf(kind, params, mode, pos_weight, T, seed=SEED):
    p = dict(params)
    if kind == "rf":
        if mode == "weight":
            p["class_weight"] = "balanced_subsample"
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **p)
    if mode == "weight":
        p["scale_pos_weight"] = pos_weight
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                         tree_method="hist", device=getattr(T, "XGB_DEVICE", "cpu"),
                         random_state=seed, n_jobs=-1, **p)


def resample(X, y, mode, seed=SEED):
    if mode != "adasyn":
        return X, y
    if not _HAS_ADASYN:
        raise ImportError("ADASYN needs imbalanced-learn: "
                          "pip install imbalanced-learn")
    # n_jobs was deprecated in imbalanced-learn 0.10 and removed in 0.12
    kw = {"random_state": seed}
    if "n_jobs" in inspect.signature(ADASYN.__init__).parameters:
        kw["n_jobs"] = -1
    Xr, yr = ADASYN(**kw).fit_resample(X, y)
    print(f"      ADASYN: {len(X):,} -> {len(Xr):,} rows "
          f"(positives {int(y.sum()):,} -> {int(yr.sum()):,})")
    return Xr, yr


# =============================================================================
# METRICS
# =============================================================================
def hour_metrics(y_true, y_pred, score=None):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and prec + rec else 0.0
    return {"F1": f1, "Recall": rec, "FDR": 1 - prec if tp + fp else np.nan,
            "Precision": prec,
            "PR_AUC": (average_precision_score(y_true, score)
                       if score is not None else np.nan),
            "POD": rec, "FAR": 1 - prec if tp + fp else np.nan,
            "CSI": tp / (tp + fp + fn) if tp + fp + fn else np.nan,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def event_metrics(ts, y_true, y_pred, tol_h=EVENT_TOLERANCE_H):
    """Fraction of true episodes flagged at least once near their onset, and
    the fraction of predicted runs that overlap a true episode."""
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    hours = (np.asarray(ts, "datetime64[h]")
             - np.datetime64(ts[0], "h")).astype(int)

    t_starts, t_lens = run_lengths(y_true)
    hit = 0
    for s, L in zip(t_starts, t_lens):
        lo, hi = hours[s] - tol_h, hours[s + L - 1] + tol_h
        if y_pred[(hours >= lo) & (hours <= hi)].any():
            hit += 1

    p_starts, p_lens = run_lengths(y_pred)
    valid = sum(1 for s, L in zip(p_starts, p_lens)
                if y_true[max(0, s - tol_h):s + L + tol_h].any())
    return {"events_true": int(len(t_lens)), "events_detected": hit,
            "event_POD": hit / len(t_lens) if len(t_lens) else np.nan,
            "alarms_raised": int(len(p_lens)),
            "alarm_precision": valid / len(p_lens) if len(p_lens) else np.nan}


def pick_threshold(y_val, p_val, grid=THRESHOLD_GRID):
    """Decision threshold maximising F1 on held-out TRAINING blocks."""
    best, best_f1 = 0.5, -1.0
    for t in grid:
        m = hour_metrics(y_val, p_val >= t)
        if m["F1"] > best_f1:
            best, best_f1 = float(t), m["F1"]
    return best, best_f1


# =============================================================================
# FIGURES
# =============================================================================
def fig_pr(curves, path, run):
    hs = sorted({h for h, _, _ in curves})
    fig, axes = plt.subplots(1, len(hs), figsize=(5.6 * len(hs), 4.6),
                             squeeze=False)
    for ax, h in zip(axes[0], hs):
        _ax(ax, axis="both")
        base = None
        for (hh, kind, mode), (rec, prec, ap, baserate) in curves.items():
            if hh != h:
                continue
            base = baserate
            ax.plot(rec, prec, color=HUE[kind], ls=MODE_STYLE[mode], lw=1.9,
                    label=f"{kind} ({mode})  AP={ap:.3f}", zorder=3)
        if base is not None:
            ax.axhline(base, color=P["axis"], lw=1.2, ls=(0, (4, 3)), zorder=2)
            ax.text(0.99, base, f" no-skill {base:.3f}", ha="right", va="bottom",
                    fontsize=8, color=P["muted"], transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_title(f"h = {h}", loc="left", pad=10)
        ax.legend(labelcolor=P["ink2"], fontsize=8, loc="lower left")
    fig.suptitle("Event classification: precision-recall", x=0.005, ha="left",
                 fontsize=12, fontweight="semibold", color=P["ink"])
    fig.text(0.005, 0.925, run, fontsize=9, color=P["muted"], ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); _show(path)


def _break_gaps(ts, v, max_gap_h=MAX_GAP_H):
    """Insert NaN where consecutive test hours are more than max_gap_h apart, so
    the line is not drawn across blocks the split assigned to training."""
    ts = pd.DatetimeIndex(ts)
    v = np.asarray(v, float)
    if len(ts) < 2:
        return ts, v
    # cast to hours explicitly: the index resolution may be ns, us or ms
    hours = ts.to_numpy().astype("datetime64[h]").astype("int64")
    cut = np.where(np.diff(hours) > max_gap_h)[0]
    if not len(cut):
        return ts, v
    ins = (ts[cut] + pd.Timedelta(hours=1)).to_numpy()
    all_ts = np.concatenate([ts.to_numpy(), ins])
    all_v = np.concatenate([v, np.full(len(cut), np.nan)])
    o = np.argsort(all_ts, kind="stable")
    return pd.DatetimeIndex(all_ts[o]), all_v[o]


def fig_confusion_series(panels, path, run, start=PLOT_START, end=PLOT_END):
    """Observed turbidity with every test hour classified TP / TN / FP / FN,
    the companion regression forecast overlaid, one panel per horizon."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    hs = sorted(panels)
    fig, axes = plt.subplots(len(hs), 1, figsize=(12.6, 4.3 * len(hs)),
                             sharex=True, squeeze=False)

    for ax, h in zip(axes[:, 0], hs):
        d = panels[h]
        ts = pd.DatetimeIndex(d["ts"])
        m = (ts >= start) & (ts <= end)
        _ax(ax)
        if not m.any():
            ax.text(0.5, 0.5, f"h = {h}: no test hours in this window",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color=P["muted"])
            continue

        t, obs = ts[m], np.asarray(d["obs"], float)[m]
        yt = np.asarray(d["y_true"], bool)[m]
        yp = np.asarray(d["y_pred"], bool)[m]

        tg, og = _break_gaps(t, obs)
        ax.plot(tg, og, color=P["ink"], lw=1.2, alpha=0.55,
                label="observed turbidity", zorder=2)
        if d.get("reg") is not None:
            _, rg = _break_gaps(t, np.asarray(d["reg"], float)[m])
            # generic label: the shared legend is built from the first panel,
            # and the family can differ between horizons
            ax.plot(tg, rg, color=P["series"][1], lw=1.5, ls=(0, (5, 2)),
                    label="regression forecast", zorder=4)

        cat = {"TP": yt & yp, "TN": ~yt & ~yp, "FP": ~yt & yp, "FN": yt & ~yp}
        n = {}
        for k in ("TN", "TP", "FP", "FN"):
            sel = cat[k]
            n[k] = int(sel.sum())
            if not sel.any():
                continue
            s = CONF[k]
            ax.scatter(t[sel], obs[sel], s=s["s"], c=s["c"], marker=s["m"],
                       linewidths=0, zorder=s["z"], label=s["lab"])

        ax.axhline(d["threshold"], color=P["axis"], lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(t[-1], d["threshold"], f"event threshold {d['threshold']:.1f} FTU ",
                va="bottom", ha="right", fontsize=8, color=P["muted"])
        ax.set_ylabel("turbidity (FTU)")
        reg_txt = f"    dashed: {d['reg_label']}" if d.get("reg") is not None else ""
        ax.set_title(f"h = {h} — classifier: {d['label']}", loc="left", pad=20)
        ax.text(0, 1.015,
                f"F1 {d['F1']:.3f} · recall {d['Recall']:.3f} · "
                f"FDR {d['FDR']:.3f}    TP {n['TP']} / FP {n['FP']} / "
                f"FN {n['FN']} / TN {n['TN']}    {int(m.sum()):,} test hours "
                f"in window{reg_txt}",
                transform=ax.transAxes, fontsize=8.5, color=P["ink2"])

    axes[-1, 0].set_xlabel("date")
    h0 = axes[0, 0].get_legend_handles_labels()
    fig.legend(*h0, loc="lower center", ncol=6, labelcolor=P["ink2"],
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    # header offsets in inches, so they hold for any number of panels
    fh = fig.get_figheight()
    fig.suptitle("Event classification against observed turbidity",
                 x=0.005, y=1 - 0.22 / fh, ha="left", fontsize=12,
                 fontweight="semibold", color=P["ink"])
    fig.text(0.005, 1 - 0.46 / fh,
             f"{run}   {start:%d %b %Y} – {end:%d %b %Y}",
             fontsize=9, color=P["muted"], ha="left", va="top")
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0.015, 1, 1 - 0.62 / fh])
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); _show(path)


# =============================================================================
# MAIN
# =============================================================================
def run(horizons=HORIZONS, models=("rf", "xgb"), modes=MODES,
        block_days=BLOCK_DAYS, train_module=TRAIN_MODULE,
        tune_threshold=TUNE_THRESHOLD):
    t0 = time.time()
    _rc()
    T = importlib.import_module(train_module)
    importlib.reload(T)

    modes = tuple(m for m in modes if m != "adasyn" or _HAS_ADASYN)
    if "adasyn" in MODES and not _HAS_ADASYN:
        print("NOTE: imbalanced-learn not installed; skipping the ADASYN mode.\n"
              "      pip install imbalanced-learn\n")

    nf = len(T.FEATURE_COLUMNS)
    T.SPLIT_MODE, T.BLOCK_DAYS = "block", block_days
    run_name = f"clf_{T.split_tag()}_{T.tree_tag()}_f{nf}_P{PERCENTILE:.0f}_r{MIN_RUN}"
    out_dir = os.path.join(T.save_root(), "events")
    os.makedirs(out_dir, exist_ok=True)

    print(f"FEATURE_COLUMNS ({nf}): {list(T.FEATURE_COLUMNS)}")
    print(f"tree lags: {T.tree_tag()}   horizons: {horizons}")
    print(f"event: > P{PERCENTILE:.0f} for >= {MIN_RUN} consecutive hours")
    print(f"split: {block_days}-day blocks   modes: {modes}")

    rows, curves, ctxs = [], {}, {}
    clf_pred, reg_pred = {}, {}          # (h, kind, mode) -> pred ; (h, kind) -> FTU
    best_fit = {}                        # h -> the best estimator so far, by F1

    for h in horizons:
        print("\n" + "=" * 86)
        print(f"HORIZON {h}")
        print("=" * 86)
        ctx = build(T, h, block_days)
        ctxs[h] = ctx
        X_tr, y_tr, X_te, y_te = ctx["X_tr"], ctx["y_tr"], ctx["X_te"], ctx["y_te"]
        pos_w = float((~y_tr).sum() / max(y_tr.sum(), 1))
        print(f"  implied scale_pos_weight = {pos_w:.2f}")

        # ---- baseline: is turbidity above the threshold at the forecast origin
        base_pred = ctx["persist_te"] > ctx["threshold"]
        m = hour_metrics(y_te, base_pred)
        m.update(event_metrics(ctx["ts_te"], y_te, base_pred))
        rows.append({"horizon": h, "model": "persistence", "mode": "-",
                     "threshold": np.nan, **m})

        fit_i, val_i = T.holdout_indices(len(X_tr), ctx["groups_tr"],
                                         VAL_FRACTION, seed=7)
        print(f"  threshold tuning: {len(fit_i):,} fit / {len(val_i):,} "
              f"validation windows (whole blocks)")

        # ---- companion regression: one model for this horizon ---------------
        if FIT_REGRESSOR:
            rk = SAVED_REGRESSORS.get(h)
            saved = None
            if rk is not None:
                print(f"\n  --- {rk.upper()} regression (already trained, "
                      f"{PARAMS_SPLIT_TAG}) ---")
                saved = load_saved_regressor(T, rk, h, nf, T.tree_tag())

            if saved is not None:
                mdl, sc, srun = saved
                p_reg = predict_saved_regressor(mdl, sc, ctx, T)
                src, lab = "saved", f"{rk.upper()} regression ({PARAMS_SPLIT_TAG})"
                ov = (random_train_overlap(T, h, ctx["ts_te"], block_days)
                      if REPORT_OVERLAP else np.nan)
                if REPORT_OVERLAP:
                    print(f"    IN-SAMPLE WARNING: {100 * ov:.1f}% of these "
                          f"{len(y_te):,} block-split test hours were in the "
                          f"{PARAMS_SPLIT_TAG} TRAINING set of this model.")
                    print("    Its scores below are therefore optimistic and are "
                          "NOT comparable with the classifiers.")
            else:
                rk = rk or models[0]
                print(f"\n  --- {rk.upper()} regression (refitted on the "
                      f"block split) ---")
                p_reg = fit_regressor(
                    rk, load_params(T, rk, h, nf, T.tree_tag(),
                                    for_classifier=False), ctx)
                src, lab, ov = "refit", f"{rk.upper()} regression", 0.0

            reg_pred[h] = {"ftu": p_reg, "label": lab, "source": src,
                           "kind": rk, "overlap": ov}
            reg_ev = p_reg > ctx["threshold"]
            m = hour_metrics(y_te, reg_ev)
            m.update(event_metrics(ctx["ts_te"], y_te, reg_ev))
            rows.append({"horizon": h,
                         "model": f"{rk.upper()} regression"
                                  + (" [in-sample]" if src == "saved" else ""),
                         "mode": "threshold", "threshold": ctx["threshold"], **m})
            rmse = float(np.sqrt(np.mean((p_reg - ctx["ftu_te"]) ** 2)))
            print(f"    RMSE {rmse:.3f} FTU;  thresholded as a classifier: "
                  f"F1 {m['F1']:.3f}  Recall {m['Recall']:.3f}  "
                  f"FDR {m['FDR']:.3f}")

        for kind in models:
            params = load_params(T, kind, h, nf, T.tree_tag())
            for mode in modes:
                print(f"\n  --- {kind.upper()} / {mode} ---")
                tag = f"{kind.upper()} ({mode})"

                # 1) threshold on held-out training blocks
                thr_d = 0.5
                if tune_threshold:
                    Xf, yf = resample(X_tr[fit_i], y_tr[fit_i], mode)
                    c = make_clf(kind, params, mode, pos_w, T)
                    c.fit(Xf, yf)
                    p_val = c.predict_proba(X_tr[val_i])[:, 1]
                    thr_d, f1v = pick_threshold(y_tr[val_i], p_val)
                    print(f"      decision threshold {thr_d:.2f} "
                          f"(validation F1 {f1v:.3f})")

                # 2) refit on ALL training rows, apply that threshold
                Xa, ya = resample(X_tr, y_tr, mode)
                c = make_clf(kind, params, mode, pos_w, T)
                c.fit(Xa, ya)
                score = c.predict_proba(X_te)[:, 1]
                pred = score >= thr_d

                m = hour_metrics(y_te, pred, score)
                m.update(event_metrics(ctx["ts_te"], y_te, pred))
                rows.append({"horizon": h, "model": kind.upper(), "mode": mode,
                             "threshold": thr_d, **m})
                print(f"      F1 {m['F1']:.3f}   Recall {m['Recall']:.3f}   "
                      f"FDR {m['FDR']:.3f}   PR-AUC {m['PR_AUC']:.3f}   "
                      f"events {m['events_detected']}/{m['events_true']}")

                prec, rec, _ = precision_recall_curve(y_te, score)
                curves[(h, kind.upper(), mode)] = (
                    rec, prec, m["PR_AUC"], float(y_te.mean()))
                clf_pred[(h, kind, mode)] = (pred, score)

                # keep only the leader: one fitted classifier per horizon in RAM
                if SAVE_MODELS and m["F1"] > best_fit.get(h, (-1.0,))[0]:
                    best_fit[h] = (m["F1"], c, kind, mode, thr_d,
                                   dict(params), dict(m))
                else:
                    del c

    # ---- tables -------------------------------------------------------------
    tab = pd.DataFrame(rows)
    cols = ["horizon", "model", "mode", "threshold", "F1", "Recall", "FDR",
            "Precision", "PR_AUC", "CSI", "TP", "FP", "FN", "TN",
            "events_true", "events_detected", "event_POD",
            "alarms_raised", "alarm_precision"]
    tab = tab[cols].sort_values(["horizon", "F1"], ascending=[True, False])
    tab.to_csv(os.path.join(out_dir, f"{run_name}.csv"), index=False)

    print("\n" + "=" * 110)
    print(f"EVENT CLASSIFICATION   [{run_name}]")
    print("=" * 110)
    print(tab.round(4).to_string(index=False))
    print("\nFDR = FP/(FP+TP) = 1 - precision.  event_POD = episodes flagged "
          f"within +/-{EVENT_TOLERANCE_H} h of onset.")

    fig_pr(curves, os.path.join(out_dir, f"fig_pr_{run_name}.png"), run_name)

    # ---- best classifier per horizon -> predictions CSV + confusion figure --
    panels = {}
    clf_only = tab[tab["mode"].isin(modes)]
    for h in horizons:
        s = clf_only[clf_only["horizon"] == h].sort_values("F1", ascending=False)
        if s.empty:
            continue
        best = s.iloc[0]
        kind = best["model"].lower()
        pred, score = clf_pred[(h, kind, best["mode"])]
        ctx = ctxs[h]
        rinfo = reg_pred.get(h)
        reg = rinfo["ftu"] if rinfo else None
        print(f"\n  h={h}: best classifier {best['model']} ({best['mode']}), "
              f"F1 {best['F1']:.3f}")

        if SAVE_MODELS and h in best_fit:
            _, est, bk, bm, thr_d, bp, bmet = best_fit[h]
            save_bundle(os.path.join(
                out_dir, f"best_classifier_h{h}_{bk}_{bm}_{run_name}.joblib"),
                est, ctx, bk, bm, h, thr_d, bp, bmet)

        out = pd.DataFrame({
            "timestamp": ctx["ts_te"], "sensor_turbidity": ctx["ftu_te"],
            "turb_event": ctx["y_te"].astype(int),
            "clf_pred": pred.astype(int), "clf_score": score})
        if reg is not None:
            out["reg_pred_FTU"] = reg
            out["reg_event"] = (reg > ctx["threshold"]).astype(int)
        out.to_csv(os.path.join(
            out_dir, f"predictions_h{h}_{run_name}.csv"), index=False)

        w = ((pd.DatetimeIndex(ctx["ts_te"]) >= pd.Timestamp(PLOT_START))
             & (pd.DatetimeIndex(ctx["ts_te"]) <= pd.Timestamp(PLOT_END)))
        span = (pd.Timestamp(PLOT_END) - pd.Timestamp(PLOT_START)) / pd.Timedelta("1h")
        print(f"       {PLOT_START} to {PLOT_END}: {int(w.sum()):,} of {span:.0f} "
              f"hours are in the test set ({100 * w.sum() / span:.0f}%)")

        panels[h] = {"ts": ctx["ts_te"], "obs": ctx["ftu_te"],
                     "y_true": ctx["y_te"], "y_pred": pred, "reg": reg,
                     "reg_label": rinfo["label"] if rinfo else "",
                     "threshold": ctx["threshold"],
                     "label": f"{best['model']} ({best['mode']})",
                     "F1": best["F1"], "Recall": best["Recall"],
                     "FDR": best["FDR"]}

    if panels:
        fig_confusion_series(
            panels, os.path.join(out_dir, f"fig_confusion_{run_name}.png"),
            run_name)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min -> {out_dir}")
    return {"table": tab, "curves": curves, "ctxs": ctxs, "panels": panels,
            "run": run_name}