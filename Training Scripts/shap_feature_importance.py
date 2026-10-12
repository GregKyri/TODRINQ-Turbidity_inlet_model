"""## **SHAP ANALYSIS**"""

"""
SHAP attribution for RF, XGBoost and LSTM -- Google Colab version.

Self-contained: the only thing it needs from your project is the training
module (for load_data / make_windows / the split and scaler functions and the
LSTMModel class). Nothing is retrained; the saved models are loaded and the
test set is rebuilt from the same fixed seeds.

USAGE IN COLAB
--------------
    # cell 1 -- setup (mount, copy off the Drive mount, cd, sys.path)
    # cell 2:
    !pip -q install shap
    import shap_colab
    res = shap_colab.run(horizon=3, window=24, split="random")

`run()` takes keyword arguments -- there is no argparse, which does not work in
a notebook because sys.argv carries the kernel's own arguments.

WHAT IT PRODUCES  (under <save_root>/shap/)
    shap_table_<run>.csv        mean |SHAP| per base feature, per model
    shap_table_lags_<run>.csv   per (feature, lag) column
    fig_shap_importance_<run>.png
    fig_shap_beeswarm_<model>_<run>.png
    fig_shap_lagprofile_<run>.png      (the informative one at window=24)
Figures are displayed inline as well as saved.

READING THE NUMBERS
-------------------
SHAP values are in the model's output space (scaled, log1p-FTU if the training
run used LOG_TARGET), not FTU. Absolute magnitudes are not comparable across
models. Use `share_pct` -- each feature's mean |SHAP| as a percentage of that
model's total, which is scale-free.

Tree SHAP is exact. The LSTM uses GradientExplainer, which is an approximation
(expected gradients), so LSTM shares carry more uncertainty than the tree ones.
Say so if you put all three in the same figure.
"""

import glob
import importlib
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.simplefilter("ignore", UserWarning)

try:
    from IPython.display import Image, display
    _IN_NOTEBOOK = True
except Exception:
    _IN_NOTEBOOK = False


# =============================================================================
# CONFIG
# =============================================================================
TRAIN_MODULE = "new_version_training"   # <- your training script, no .py

N_TREE_SAMPLES = 500       # test rows explained for RF / XGB (exact, cheap)
N_LSTM_SAMPLES = 300        # test rows for the LSTM (approximate, slow on CPU)
N_BACKGROUND = 100          # background rows for GradientExplainer
LSTM_NSAMPLES = 100         # gradient samples per explained row

TOP_N_BEESWARM = 20
TOP_N_BAR = 12
TOP_N_LAGLINES = 7
FIG_DPI = 200
DARK = False


# =============================================================================
# PALETTE  (validated categorical order, recessive chrome)
# =============================================================================
LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
         "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
         "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                    "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
         "div": ["#2a78d6", "#f0efec", "#d03b3b"]}
DARKP = {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
         "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
         "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
                    "#d55181", "#008300", "#9085e9", "#e66767"],
         "div": ["#3987e5", "#383835", "#d03b3b"]}


def P():
    return DARKP if DARK else LIGHT


def apply_rc():
    p = P()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "figure.facecolor": p["surface"], "savefig.facecolor": p["surface"],
        "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.titlecolor": p["ink"], "axes.labelcolor": p["ink2"],
        "axes.labelsize": 10, "text.color": p["ink"],
        "legend.frameon": False, "legend.fontsize": 9,
    })


def style_axes(ax, axis="x"):
    p = P()
    ax.set_facecolor(p["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(p["axis"]); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=p["muted"], labelsize=9, length=0)
    ax.grid(axis=axis, color=p["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    return ax


def show(path):
    print(f"  -> {path}")
    if _IN_NOTEBOOK:
        display(Image(path))


# =============================================================================
# REBUILD THE TEST SET (identical seeds -> identical rows)
# =============================================================================
def rebuild(T, split, block_days, horizon, window):
    T.SPLIT_MODE, T.BLOCK_DAYS = split, block_days
    T.INPUT_WINDOW, T.HORIZONS = window, [horizon]

    df, valid = T.load_data()
    feats = list(T.FEATURE_COLUMNS)

    out = T.make_windows(df, valid, feats, input_window=window,
                         forecast_horizon=horizon)
    X_all, y_all, ts_all, spans = out[0], out[1], out[2], out[3]

    row_block = T.row_block_ids(df.index, block_days)
    idx_tr, idx_te, groups_tr = T.split_windows(X_all, y_all, ts_all,
                                                spans, row_block)

    scalers = T.fit_window_scalers(X_all[idx_tr], y_all[idx_tr], feats, window)
    X_tr, _ = T.apply_window_scalers(X_all[idx_tr], y_all[idx_tr],
                                     scalers, feats, window)
    X_te, y_te = T.apply_window_scalers(X_all[idx_te], y_all[idx_te],
                                        scalers, feats, window)

    print(f"\n  train windows {X_tr.shape}   test windows {X_te.shape}")
    print("  CHECK: the test count above must match the n in your results "
          "table for this scenario. If it does not, the split settings are "
          "wrong and the SHAP values will not correspond to the saved model.")

    return dict(feature_columns=feats,
                flat_names=T.expanded_feature_names(feats, window),
                X_tr=X_tr, X_te=X_te, y_te=y_te, window=window,
                horizon=horizon, n_features=len(feats))


def subsample(X, n, seed=0):
    if len(X) <= n:
        return X
    rng = np.random.default_rng(seed)
    return X[np.sort(rng.choice(len(X), n, replace=False))]


# =============================================================================
# MODEL LOADING  (glob, so it works whatever the run-name convention)
# =============================================================================
def _find(root, folder, pattern):
    hits = sorted(glob.glob(os.path.join(root, folder, "*", pattern)))
    if not hits:
        seen = sorted(glob.glob(os.path.join(root, folder, "*", "*")))
        raise FileNotFoundError(
            f"no file matching {pattern!r} under {root}/{folder}/*\n"
            f"  what is there: {seen[:12]}")
    if len(hits) > 1:
        print(f"    note: {len(hits)} matches, using {hits[0]}")
    return hits[0]


def load_tree(root, kind, window, horizon):
    folder = {"rf": "RF models", "xgb": "XGB models"}[kind]
    path = _find(root, folder, f"{kind}_*w{window}_h{horizon}.joblib")
    print(f"  {kind.upper()}: {path}")
    return joblib.load(path)


def load_lstm(root, T, window, horizon, n_features):
    """Architecture is inferred from the state_dict, so no config file needed."""
    import torch
    path = _find(root, "LSTM models", f"lstm_*w{window}_h{horizon}.pth")
    print(f"  LSTM: {path}")
    sd = torch.load(path, map_location="cpu")

    w_ih = sd["lstm.weight_ih_l0"]
    hidden = w_ih.shape[0] // 4
    in_size = w_ih.shape[1]
    n_layers = sum(1 for k in sd if k.startswith("lstm.weight_ih_l"))
    print(f"        inferred: hidden={hidden}, layers={n_layers}, "
          f"input_size={in_size}")
    if in_size != n_features:
        raise ValueError(
            f"model expects {in_size} features, the rebuilt data has "
            f"{n_features}. FEATURE_COLUMNS does not match the trained model.")

    model = T.LSTMModel(n_features, hidden, n_layers, 0.0)
    model.load_state_dict(sd)
    model.eval()
    return model


# =============================================================================
# EXPLAINERS
# =============================================================================
def shap_tree(model, X, kind):
    import shap
    print(f"  {kind.upper()}: TreeExplainer on {len(X):,} rows (exact) ...")
    ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    try:
        v = ex.shap_values(X, check_additivity=False)
    except TypeError:
        v = ex.shap_values(X)
    v = np.asarray(v)
    return v[..., 0] if v.ndim == 3 else v


def shap_lstm(model, X_bg, X_expl, window, n_features):
    """GradientExplainer on CPU.

    DeepExplainer's rules do not cover fused LSTM kernels, so the gradient
    explainer is the one that works on a recurrent net, and it runs on CPU
    because the fused GPU kernel does not support the double backward pass.
    Returns None (and says so) rather than substituting something that is not
    SHAP.
    """
    import torch, shap
    bg = torch.from_numpy(X_bg.reshape(-1, window, n_features)).float()
    te = torch.from_numpy(X_expl.reshape(-1, window, n_features)).float()
    print(f"  LSTM: GradientExplainer on {len(X_expl):,} rows "
          f"(background {len(bg):,}, {LSTM_NSAMPLES} samples/row, CPU) ...")
    print("        this is the slow step -- a few minutes is normal")
    try:
        ex = shap.GradientExplainer(model, bg)
        v = ex.shap_values(te, nsamples=LSTM_NSAMPLES)
    except Exception as e:
        print(f"  LSTM: GradientExplainer FAILED ({type(e).__name__}: {e})")
        print("        falling back to permutation importance -- reported "
              "separately, NOT mixed into the SHAP table")
        return None
    if isinstance(v, list):
        v = v[0]
    v = np.asarray(v)
    if v.ndim == 4:
        v = v[..., 0]
    return v.reshape(len(X_expl), -1)     # ravel order t-W ... t-1


def permutation_importance_lstm(model, X, window, n_features, n_repeats=3,
                                seed=0):
    """Model-agnostic fallback. NOT SHAP -- keep it in its own table."""
    import torch
    rng = np.random.default_rng(seed)

    def pred(A):
        with torch.no_grad():
            t = torch.from_numpy(A.reshape(-1, window, n_features)).float()
            return model(t).numpy().ravel()

    base = pred(X)
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        d = 0.0
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            d += np.mean((pred(Xp) - base) ** 2)
        imp[j] = d / n_repeats
    return imp


# =============================================================================
# AGGREGATION
# =============================================================================
def aggregate(shap_by_model, flat_names, feature_columns, window, horizon):
    """Adds BOTH lag conventions so neither is ambiguous:
         lag_from_origin_h : hours before the forecast origin  (1 .. W)
         lag_to_target_h   : hours before the predicted value  (h .. h+W-1)
    """
    n_feat = len(feature_columns)
    lag_rows, base_rows = [], []
    for model, vals in shap_by_model.items():
        mabs = np.abs(vals).mean(axis=0)
        total = mabs.sum()
        for j, nm in enumerate(flat_names):
            blk = j // n_feat                 # 0 = oldest row in the window
            lag_origin = window - blk
            lag_target = lag_origin + horizon - 1
            lag_rows.append({"model": model, "column": nm,
                             "feature": feature_columns[j % n_feat],
                             "lag_from_origin_h": lag_origin,
                             "lag_to_target_h": lag_target,
                             "mean_abs_shap": float(mabs[j]),
                             "share_pct": float(100 * mabs[j] / total)})
        per_feat = mabs.reshape(window, n_feat).sum(axis=0)
        for k, f in enumerate(feature_columns):
            base_rows.append({"model": model, "feature": f,
                              "mean_abs_shap": float(per_feat[k]),
                              "share_pct": float(100 * per_feat[k] / total)})

    lag_df, base_df = pd.DataFrame(lag_rows), pd.DataFrame(base_rows)
    wide = base_df.pivot(index="feature", columns="model", values="share_pct")
    wide["mean_share_pct"] = wide.mean(axis=1)
    return lag_df, base_df, wide.sort_values("mean_share_pct", ascending=False)

# =============================================================================
# FIGURES
# =============================================================================
def fig_importance_bar(wide, path, run):
    p = P()
    models = [c for c in wide.columns if c != "mean_share_pct"]
    d = wide.head(TOP_N_BAR).iloc[::-1]
    feats, n_m = list(d.index), len(models)

    fig, ax = plt.subplots(figsize=(8.4, max(3.4, 0.30 * len(feats) * n_m + 1.4)))
    style_axes(ax, "x")
    y = np.arange(len(feats)); bh = 0.8 / n_m
    for i, m in enumerate(models):
        v = d[m].to_numpy()
        ax.barh(y + (i - (n_m - 1) / 2) * bh, v, height=bh * 0.9, label=m,
                color=p["series"][i], edgecolor=p["surface"], linewidth=1.0,
                zorder=3)
        for yy, vv in zip(y + (i - (n_m - 1) / 2) * bh, v):
            if np.isfinite(vv) and vv > 0.8:
                ax.text(vv + 0.4, yy, f"{vv:.1f}", va="center", fontsize=8,
                        color=p["ink2"])
    ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=9, color=p["ink2"])
    ax.set_xlabel("share of total |SHAP| (%)")
    ax.set_title("Feature attribution by model", loc="left", pad=12)
    ax.text(0, 1.02, f"{run} · summed over all 24 input lags",
            transform=ax.transAxes, fontsize=9, color=p["muted"])
    if n_m >= 2:
        ax.legend(loc="lower right", labelcolor=p["ink2"])
    fig.tight_layout(); fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig); show(path)


def fig_beeswarm(vals, X, flat_names, model, path, run):
    import shap
    p = P()
    cmap = LinearSegmentedColormap.from_list("div", p["div"], N=256)
    plt.figure(figsize=(8.4, 0.34 * TOP_N_BEESWARM + 2.0))
    shap.summary_plot(vals, X, feature_names=flat_names, show=False,
                      max_display=TOP_N_BEESWARM, cmap=cmap)
    ax = plt.gca(); style_axes(ax, "x")
    ax.set_xlabel("SHAP value (model output space)")
    ax.set_title(f"{model} — per-sample attribution", loc="left", pad=12)
    ax.text(0, 1.015, run, transform=ax.transAxes, fontsize=9, color=p["muted"])
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close(); show(path)


def fig_lag_profile(lag_df, path, run, horizon, window):
    p = P()
    models = sorted(lag_df["model"].unique())
    order = (lag_df.groupby("feature")["share_pct"].sum()
             .sort_values(ascending=False).index.tolist()[:TOP_N_LAGLINES])
    lo, hi = horizon, horizon + window - 1
    fig, axes = plt.subplots(1, len(models), figsize=(4.9 * len(models), 4.0),
                             sharey=True, squeeze=False)
    for ai, m in enumerate(models):
        ax = style_axes(axes[0][ai], "y")
        sub = lag_df[lag_df["model"] == m]
        for k, f in enumerate(order):
            s = sub[sub["feature"] == f].sort_values("lag_to_target_h")
            ax.plot(s["lag_to_target_h"], s["share_pct"], linewidth=2.0,
                    label=f, color=p["series"][k % 8],
                    solid_capstyle="round", zorder=3)

        ax.axvspan(0, lo, color=p["grid"], alpha=0.55, lw=0, zorder=1)
        ax.axvline(lo, color=p["axis"], lw=1.0, ls=(0, (4, 3)), zorder=2)
        if ai == 0:
            ax.text(lo / 2, ax.get_ylim()[1] * 0.96,
                    f"{horizon} h\nlead", ha="center", va="top",
                    fontsize=8, color=p["muted"], linespacing=1.2)

        ax.set_xlim(hi + 0.7, 0)            # target at the right, past to the left
        ax.set_xlabel("hours before the predicted value")
        if ai == 0:
            ax.set_ylabel("share of total |SHAP| (%)")
        ax.set_title(m, loc="left", pad=8)

    axes[0][-1].legend(loc="upper left", labelcolor=P()["ink2"])
    fig.suptitle("Where each model looks in time", x=0.005, ha="left",
                 fontsize=12, fontweight="semibold", color=P()["ink"])
    fig.text(0.005, 0.93, f"{run} · inputs span t−{window-1}…t, "
             f"i.e. {lo}–{hi} h before the target",
             fontsize=9, color=P()["muted"], ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); show(path)


# =============================================================================
# MAIN
# =============================================================================
def run(horizon, window, split="random", block_days=30,
        models=("RF", "XGB"), train_module=TRAIN_MODULE,
        n_tree=N_TREE_SAMPLES, n_lstm=N_LSTM_SAMPLES, dark=False):
    global DARK
    DARK = dark
    apply_rc()

    T = importlib.import_module(train_module)
    importlib.reload(T)          # pick up edits without restarting the runtime
    T.SPLIT_MODE, T.BLOCK_DAYS = split, block_days
    root = T.save_root()
    print(f"run root: {root}")

    ctx = rebuild(T, split, block_days, horizon, window)
    out_dir = os.path.join(root, "shap"); os.makedirs(out_dir, exist_ok=True)
    run_name = f"{T.split_tag()}_w{window}_h{horizon}"

    X_tree = subsample(ctx["X_te"], n_tree, seed=0)
    X_lstm = subsample(ctx["X_te"], n_lstm, seed=0)
    X_bg = subsample(ctx["X_tr"], N_BACKGROUND, seed=1)

    shap_by_model, extras = {}, {}
    print("\nEXPLAINERS")
    for kind, label in (("rf", "RF"), ("xgb", "XGB")):
        if label in models:
            shap_by_model[label] = shap_tree(
                load_tree(root, kind, window, horizon), X_tree, kind)

    if "LSTM" in models:
        m = load_lstm(root, T, window, horizon, ctx["n_features"])
        v = shap_lstm(m, X_bg, X_lstm, window, ctx["n_features"])
        if v is not None:
            shap_by_model["LSTM"] = v
        else:
            extras["LSTM_permutation"] = permutation_importance_lstm(
                m, X_lstm, window, ctx["n_features"])

    if not shap_by_model:
        raise SystemExit("nothing explained")

    lag_df, base_df, wide = aggregate(shap_by_model, ctx["flat_names"],
                                      ctx["feature_columns"], window,horizon)

    t1 = os.path.join(out_dir, f"shap_table_{run_name}.csv")
    t2 = os.path.join(out_dir, f"shap_table_lags_{run_name}.csv")
    wide.round(4).to_csv(t1); lag_df.round(6).to_csv(t2, index=False)
    print(f"\n  -> {t1}\n  -> {t2}")

    print("\n" + "=" * 70)
    print(f"SHARE OF TOTAL |SHAP| BY FEATURE (%)   [{run_name}]")
    print("=" * 70)
    print(wide.round(2).to_string())
    if "LSTM" in shap_by_model:
        print("\n  note: RF/XGB values are exact tree SHAP; LSTM values are "
              "approximate (expected gradients) and explain fewer rows.")

    print("\nFIGURES")
    fig_importance_bar(wide, os.path.join(
        out_dir, f"fig_shap_importance_{run_name}.png"), run_name)
    for m, v in shap_by_model.items():
        X = X_lstm if m == "LSTM" else X_tree
        fig_beeswarm(v, X, ctx["flat_names"], m, os.path.join(
            out_dir, f"fig_shap_beeswarm_{m}_{run_name}.png"), run_name)
    if window > 1:
        lag_path = os.path.join(out_dir, f"fig_shap_lagprofile_{run_name}.png")
        fig_lag_profile(lag_df, lag_path, run_name, horizon, window)

    if extras:
        pi = pd.DataFrame({
            "column": ctx["flat_names"],
            "permutation_importance": extras["LSTM_permutation"]}
        ).sort_values("permutation_importance", ascending=False)
        p3 = os.path.join(out_dir, f"lstm_permutation_{run_name}.csv")
        pi.to_csv(p3, index=False)
        print(f"\n  LSTM permutation importance (fallback) -> {p3}")

    print("\nDone.")
    return {"wide": wide, "lag_df": lag_df, "shap": shap_by_model,
            "ctx": ctx, "out_dir": out_dir}