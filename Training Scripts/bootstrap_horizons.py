"""
Multi-horizon bootstrap-ensemble UQ for RF and XGBoost, h = 1 ... 6.

Reuses every primitive from the single-horizon script (load_params, fit_members,
fit_single, draw_with_resid, crps_ensemble, interval_stats, variance_budget,
fig_series), so the protocol is identical. That script is not modified.

WHAT IT PRODUCES
    fig_h_skill_<run>.png        RMSE vs horizon | CRPS vs horizon, all models
    fig_h_series_h<h>_<run>.png  observations, 90% band, deterministic models
    fig_h_scatter_<run>.png      predicted vs observed for the best model
    multi_horizon_uq_<run>.csv   every metric, every model, every horizon
    multi_horizon_budget_<run>.csv  parameter / structural / unexplained by horizon

Reliability, PIT and coverage-by-quintile are deliberately NOT computed.

-----------------------------------------------------------------------------
WHY THE SPLIT IS REBUILT HERE RATHER THAN TAKEN FROM bootstrap_uq.rebuild
-----------------------------------------------------------------------------
random_split_indices() calls train_test_split(np.arange(n)) and n shrinks as
the horizon grows, because make_windows drops h-1 more windows at the end of
the record. Each horizon therefore draws an INDEPENDENT random 20% test set:
the six test sets overlap in ~0.2^6 of rows, i.e. essentially nowhere, and a
row that is test at h=1 may be training at h=6.

Two consequences, both fatal to a horizon curve:
  * RMSE at h=1 and h=6 are computed on different samples, so the curve mixes
    horizon skill with sampling noise (this is what made persistence appear to
    IMPROVE from h=4 to h=5 in the earlier table);
  * nothing stops a horizon being evaluated on rows another horizon trained on.

So the partition is defined ONCE, at the longest horizon, as a set of target
TIMESTAMPS (make_windows stamps each window with idx[i + h_off], the hour being
predicted). Every horizon then takes its train and test rows by timestamp
membership. Same hours predicted at every lead time, from origins h hours
earlier, with no row changing side. Set ALIGN_SPLITS=False only to reproduce
the old per-horizon behaviour.

USAGE (Colab)
    import multi_horizon_uq, importlib
    importlib.reload(multi_horizon_uq)
    res = multi_horizon_uq.run(split="random")             # h = 1..6, B = 10
    res = multi_horizon_uq.run(horizons=(6,), B=10)        # time one horizon

COST
    (B + 1) x 2 x len(horizons) model fits -- 132 at the defaults. Fitted
    members are CACHED to disk, so a crash, a changed residual scheme or a
    changed figure costs nothing: re-running reuses every fit it already has.
"""

import hashlib
import importlib
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:                                  # the single-horizon script, whatever it is called
    import bootstrap_uq as U
except ImportError:                   # renamed copy
    import Bootstrap as U


# =============================================================================
# CONFIG
# =============================================================================
HORIZONS = (1, 2, 3, 4, 5, 6)
ALIGN_SPLITS = True         # one partition, by target timestamp, for all horizons
COMMON_SAMPLE = True        # then evaluate on timestamps present at every horizon
MAKE_SERIES_PER_HORIZON = True
SCATTER_HORIZON = None      # None -> the longest horizon in the run
CACHE = True                # reuse fitted members across runs

RMSE_SERIES = ["persistence", "RF single", "XGB single",
               "RF+PE", "XGB+PE", "POOL+PE"]
CRPS_SERIES = ["persistence", "RF single", "XGB single", "RF+R", "XGB+R",
               "RF+PE", "XGB+PE", "POOL+PE"]

VAR_STYLE = {"single": (0, (1, 2)), "ens": (0, (3, 1, 1, 1)),
             "+R": (0, (5, 2)), "+PE": "-", "persistence": (0, (1, 2))}
VAR_MARK = {"single": "o", "ens": "^", "+R": "s", "+PE": "D",
            "persistence": "x"}


def _variant(name):
    if name == "persistence":
        return "persistence"
    for v in ("+PE", "+R"):
        if name.endswith(v):
            return v
    return "ens" if name.endswith(" ens") else "single"


def _hue(name):
    if name == "persistence":
        return U.P["muted"]
    return U._hue(name.split()[0])


# =============================================================================
# METRIC ROWS
# =============================================================================
def _point_row(h, name, y, p):
    p = np.asarray(p, float)
    mae = float(np.mean(np.abs(p - y)))
    return {"horizon": h, "model": name, "B": 1, "kind": "point",
            "RMSE": float(np.sqrt(np.mean((p - y) ** 2))), "MAE": mae,
            "CRPS_fair": mae, "CRPS_biased": mae,      # point forecast: CRPS = MAE
            "coverage90": np.nan, "width90_FTU": np.nan}


def _band_row(h, name, y, M, B):
    cov, wid, _, _ = U.interval_stats(y, M)
    mu = M.mean(axis=0)
    return {"horizon": h, "model": name, "B": B, "kind": "band",
            "RMSE": float(np.sqrt(np.mean((mu - y) ** 2))),
            "MAE": float(np.mean(np.abs(mu - y))),
            "CRPS_fair": U.crps_ensemble(y, M, fair=True),
            "CRPS_biased": U.crps_ensemble(y, M, fair=False),
            "coverage90": cov, "width90_FTU": wid}


def _nse(y, p):
    return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))


# =============================================================================
# SPLIT CONSTRUCTION  (the fix -- see module docstring)
# =============================================================================
def _windows(T, horizon, block_days):
    df, valid = T.load_data()
    feats = list(T.FEATURE_COLUMNS)
    W = T.INPUT_WINDOW
    X, y, ts, spans, pers = T.make_windows(
        df, valid, feats, input_window=W, forecast_horizon=horizon)
    return dict(feats=feats, W=W, X=X, y=y, ts=ts, spans=spans, pers=pers,
                row_block=T.row_block_ids(df.index, block_days))


def reference_partition(T, split, block_days, horizon):
    """Define train/test ONCE, as sets of target timestamps."""
    T.SPLIT_MODE, T.BLOCK_DAYS, T.HORIZONS = split, block_days, [horizon]
    b = _windows(T, horizon, block_days)
    idx_tr, idx_te, _ = T.split_windows(b["X"], b["y"], b["ts"],
                                        b["spans"], b["row_block"])
    print(f"  reference partition fixed at h = {horizon}: "
          f"{len(idx_tr):,} train / {len(idx_te):,} test target hours")
    return {"train": b["ts"][idx_tr], "test": b["ts"][idx_te]}


def rebuild_aligned(T, split, block_days, horizon, ref=None):
    """Training/test arrays for one horizon.

    With ref, rows are assigned by target-timestamp membership in the reference
    partition, so every horizon predicts the same hours and no row changes side.
    Without ref, falls back to the per-horizon split (not comparable across h).
    """
    T.SPLIT_MODE, T.BLOCK_DAYS, T.HORIZONS = split, block_days, [horizon]
    b = _windows(T, horizon, block_days)
    X, y, ts, spans, feats, W = (b["X"], b["y"], b["ts"], b["spans"],
                                 b["feats"], b["W"])

    if ref is None:
        idx_tr, idx_te, groups_tr = T.split_windows(X, y, ts, spans, b["row_block"])
    else:
        idx_tr = np.where(ts.isin(ref["train"]))[0]
        idx_te = np.where(ts.isin(ref["test"]))[0]
        win_block = b["row_block"][spans[:, 1]]
        groups_tr = win_block[idx_tr] if split == "block" else None
        drop = len(ts) - len(idx_tr) - len(idx_te)
        print(f"  aligned: {len(idx_tr):,} train / {len(idx_te):,} test"
              f"   ({drop:,} windows not in the reference partition)")

    if len(idx_te) == 0 or len(idx_tr) == 0:
        raise ValueError(
            f"h={horizon}: empty train or test set after alignment "
            f"({len(idx_tr)} / {len(idx_te)}). The reference partition does not "
            "match this horizon's windows.")

    scalers = T.fit_window_scalers(X[idx_tr], y[idx_tr], feats, W)
    X_tr, y_tr = T.apply_window_scalers(X[idx_tr], y[idx_tr], scalers, feats, W)
    X_te, _ = T.apply_window_scalers(X[idx_te], y[idx_te], scalers, feats, W)
    cols = T.tree_column_index(feats, W)

    return dict(X_tr=X_tr[:, cols], X_te=X_te[:, cols], y_tr=y_tr,
                y_te_ftu=y[idx_te].astype(float), ts_te=ts[idx_te],
                persist_te=b["pers"][idx_te].astype(float),
                groups_tr=groups_tr, scalers=scalers, T=T)


# =============================================================================
# MEMBER CACHE
# =============================================================================
def _cache_key(T, kind, horizon, split, B, nf):
    s = (f"{kind}|h{horizon}|{split}|B{B}|f{nf}|{T.tree_tag()}|"
         f"{U.SAMPLE_FRAC}|{U.REPLACE}|{U.SEED}|{ALIGN_SPLITS}|{U.MAX_OOB_RESID}")
    return f"{kind}_h{horizon}_{hashlib.md5(s.encode()).hexdigest()[:10]}.npz"


def _cache_load(path):
    if not (CACHE and os.path.exists(path)):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        n = int(z["n_resid"])
        return (z["S"], z["F"], [z[f"r{i}"] for i in range(n)],
                z["s_single"], z["p_single"], z["s_resid"])
    except Exception as e:                       # corrupt or stale -> refit
        print(f"    cache unreadable ({e}); refitting")
        return None


def _cache_save(path, S, F, resids, s_single, p_single, s_resid):
    if not CACHE:
        return
    d = {"S": S, "F": F, "n_resid": len(resids), "s_single": s_single,
         "p_single": p_single, "s_resid": s_resid}
    d.update({f"r{i}": r for i, r in enumerate(resids)})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **d)


# =============================================================================
# PREFLIGHT
# =============================================================================
def preflight(T, horizons, models, split, block_days):
    """Load every hyperparameter set and build every split BEFORE fitting, so a
    missing config or an empty sample fails in seconds rather than hours in."""
    nf = len(T.FEATURE_COLUMNS)
    tag = T.tree_tag()
    params, missing = {}, []
    print("=" * 86)
    print("PREFLIGHT: hyperparameters")
    print("=" * 86)
    for h in horizons:
        for kind in models:
            try:
                params[(h, kind)] = U.load_params(T, kind, h, nf, tag)
            except FileNotFoundError:
                missing.append(f"{kind.upper()} h={h}")
    if missing:
        raise FileNotFoundError(
            "no tuned config for: " + ", ".join(missing)
            + f"\n  (looking for {nf} features, lag tag '{tag}')"
            + "\n  Tune those scenarios first, or pass a shorter horizons=(...).")

    print("\n" + "=" * 86)
    print("PREFLIGHT: splits")
    print("=" * 86)
    ref = None
    if ALIGN_SPLITS and len(horizons) > 1:
        print(f"\nreference partition (longest horizon, h = {max(horizons)})")
        ref = reference_partition(T, split, block_days, max(horizons))
    elif len(horizons) > 1:
        print("\nALIGN_SPLITS OFF -- each horizon draws its own split. Under a "
              "random split the test sets barely overlap and the curves are "
              "NOT comparable across horizons.")

    ctxs = {}
    for h in horizons:
        print(f"\nh = {h}")
        ctxs[h] = rebuild_aligned(T, split, block_days, h, ref)
        print(f"  train {ctxs[h]['X_tr'].shape}   test {ctxs[h]['X_te'].shape}")

    masks = {h: np.ones(len(ctxs[h]["ts_te"]), bool) for h in horizons}
    if COMMON_SAMPLE and len(horizons) > 1:
        common = None
        for h in horizons:
            t = ctxs[h]["ts_te"]
            common = t if common is None else common.intersection(t)
        masks = {h: ctxs[h]["ts_te"].isin(common) for h in horizons}
        smallest = min(len(ctxs[h]["ts_te"]) for h in horizons)
        print(f"\nCOMMON SAMPLE: {len(common):,} target hours shared by all horizons")
        for h in horizons:
            n = len(ctxs[h]["ts_te"])
            print(f"  h={h}: {n:,} test rows -> {int(masks[h].sum()):,} evaluated "
                  f"({n - int(masks[h].sum()):,} dropped)")
        if len(common) == 0:
            raise ValueError(
                "the horizons share NO test timestamps. With ALIGN_SPLITS=True "
                "this should be impossible -- check that make_windows still "
                "stamps windows with the target hour.")
        if len(common) < 0.5 * smallest:
            print(f"\n  WARNING: the common sample is only "
                  f"{100 * len(common) / smallest:.0f}% of the smallest test set. "
                  "Expect it to be >95% when the splits are aligned.")
    else:
        print("\nCOMMON SAMPLE OFF -- RMSE is NOT comparable across horizons.")

    return ctxs, params, masks


# =============================================================================
# FIGURES
# =============================================================================
def fig_horizon_skill(tab, path, run, rmse_series=None, crps_series=None):
    """RMSE and CRPS against forecast horizon. Hue = family, style = variant."""
    rmse_series = rmse_series or RMSE_SERIES
    crps_series = crps_series or CRPS_SERIES
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    for ax, series, col, lab in (
            (axes[0], rmse_series, "RMSE", "RMSE (FTU)"),
            (axes[1], crps_series, "CRPS_fair", "CRPS, fair (FTU)")):
        U._ax(ax)
        for name in series:
            s = tab[tab["model"] == name].sort_values("horizon")
            if s.empty:
                continue
            v = _variant(name)
            ax.plot(s["horizon"], s[col], color=_hue(name), ls=VAR_STYLE[v],
                    marker=VAR_MARK[v], markersize=5, lw=1.9,
                    markerfacecolor=U.P["surface"] if v in ("+R", "single")
                    else None, label=name, zorder=3)
        ax.set_xlabel("forecast horizon (hours)")
        ax.set_ylabel(lab)
        ax.set_xticks(sorted(tab["horizon"].unique()))
    axes[0].set_title("Point accuracy", loc="left", pad=10)
    axes[1].set_title("Probabilistic skill", loc="left", pad=10)

    order = list(dict.fromkeys(list(crps_series) + list(rmse_series)))
    proxies = [plt.Line2D([], [], color=_hue(nm), ls=VAR_STYLE[_variant(nm)],
                          marker=VAR_MARK[_variant(nm)], markersize=5, lw=1.9,
                          markerfacecolor=U.P["surface"]
                          if _variant(nm) in ("+R", "single") else None)
               for nm in order]
    fig.legend(proxies, order, loc="lower center", ncol=min(5, len(order)),
               labelcolor=U.P["ink2"], fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Skill against forecast horizon", x=0.005, ha="left",
                 fontsize=12, fontweight="semibold", color=U.P["ink"])
    fig.text(0.005, 0.925, run, fontsize=9, color=U.P["muted"], ha="left")
    fig.tight_layout(rect=[0, 0.02, 1, 0.91])
    fig.savefig(path, dpi=U.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    U._show(path)


def fig_scatter(y, p, lo, hi, ts, name, h, path, run, window=14 * 24):
    """Left: predicted vs observed with the 1:1 line. Right: the same forecast
    in the time domain with its 90% interval, over the largest event."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        print("  scatter skipped: no evaluated rows")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    c = _hue(name)

    ax = U._ax(axes[0], axis="both")
    m = (y > 0) & (p > 0)
    lim = [max(0.5, min(y[m].min(), p[m].min()) * 0.8),
           max(y[m].max(), p[m].max()) * 1.2]
    ax.plot(lim, lim, color=U.P["axis"], lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.scatter(y[m], p[m], s=7, color=c, alpha=0.22, lw=0, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("observed turbidity (FTU)")
    ax.set_ylabel("predicted turbidity (FTU)")
    ax.set_title(f"{name}, h = {h}", loc="left", pad=10)
    ax.text(0.04, 0.96,
            f"RMSE  {np.sqrt(np.mean((p - y) ** 2)):.3f} FTU\n"
            f"MAE   {np.mean(np.abs(p - y)):.3f} FTU\n"
            f"NSE   {_nse(y, p):.3f}\n"
            f"n     {len(y):,}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color=U.P["ink2"], family="monospace")

    k = min(window, len(y))
    i = int(np.argmax(y)); a = max(0, i - k // 2); b = min(len(y), a + k)
    ax = U._ax(axes[1])
    if lo is not None:
        ax.fill_between(ts[a:b], lo[a:b], hi[a:b], color=c, alpha=0.18, lw=0,
                        label="90% interval", zorder=2)
    ax.plot(ts[a:b], p[a:b], color=c, lw=1.9, label=f"{name} predicted", zorder=4)
    ax.plot(ts[a:b], y[a:b], color=U.P["ink"], lw=1.6, label="observed", zorder=5)
    ax.set_ylabel("turbidity (FTU)")
    ax.set_title("Largest event in the test period", loc="left", pad=10)
    top = max(y[a:b].max(), hi[a:b].max() if lo is not None else 0)
    ax.set_ylim(top=top * 1.22)
    ax.legend(labelcolor=U.P["ink2"], fontsize=8, ncol=3, loc="upper left")

    fig.suptitle("Best model: predicted against observed", x=0.005, ha="left",
                 fontsize=12, fontweight="semibold", color=U.P["ink"])
    fig.text(0.005, 0.93, run, fontsize=9, color=U.P["muted"], ha="left")
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(path, dpi=U.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    U._show(path)


# =============================================================================
# MAIN
# =============================================================================
def run(horizons=HORIZONS, split="random", block_days=30, models=("rf", "xgb"),
        B=None, train_module=U.TRAIN_MODULE, scatter_horizon=SCATTER_HORIZON,
        pool_families=True):
    t_start = time.time()
    U._rc()
    T = importlib.import_module(train_module)
    importlib.reload(T)
    T.SPLIT_MODE, T.BLOCK_DAYS = split, block_days
    B = U.B_MAX if B is None else B
    horizons = tuple(sorted(horizons))

    nf = len(T.FEATURE_COLUMNS)
    frac = ("bootstrap (63.2% unique)" if U.SAMPLE_FRAC is None else
            f"{U.SAMPLE_FRAC:.0%} {'with' if U.REPLACE else 'without'} replacement")
    run_name = f"{T.split_tag()}_{T.tree_tag()}_f{nf}_B{B}"
    if U.SAMPLE_FRAC is not None:
        run_name += f"_s{int(round(100 * U.SAMPLE_FRAC))}"

    print(f"FEATURE_COLUMNS ({nf}): {list(T.FEATURE_COLUMNS)}")
    print(f"tree lags: {T.tree_tag()}   horizons: {horizons}   split: {split}")
    print(f"resampling: {frac}   B = {B}   aligned splits: {ALIGN_SPLITS}")
    print(f"model fits: {(B + 1) * len(models) * len(horizons)} (cached fits reused)")

    ctxs, params, masks = preflight(T, horizons, models, split, block_days)

    out_dir = os.path.join(T.save_root(), "uq_horizons")
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(out_dir, exist_ok=True)

    rows, budgets, store = [], [], {}

    for h in horizons:
        ctx, mk = ctxs[h], masks[h]
        y = ctx["y_te_ftu"][mk]
        ts = ctx["ts_te"][mk]
        pers = ctx["persist_te"][mk]
        print("\n" + "=" * 86)
        print(f"HORIZON {h}   ({len(y):,} evaluated rows)")
        print("=" * 86)

        rows.append(_point_row(h, "persistence", y, pers))
        store[(h, "persistence")] = {"p": pers, "lo": None, "hi": None}

        fam, fam_pe, singles_ftu = {}, {}, {}

        for kind in models:
            label = kind.upper()
            pr = params[(h, kind)]
            cpath = os.path.join(cache_dir, _cache_key(T, kind, h, split, B, nf))
            cached = _cache_load(cpath)

            if cached is not None:
                S, F, resids, s_single, p_single, s_resid = cached
                print(f"\n--- {label}: reusing cached fits "
                      f"({os.path.basename(cpath)}) ---")
            else:
                print(f"\n--- {label} single ---")
                _, s_single, p_single, s_resid = U.fit_single(kind, pr, ctx)
                print(f"\n--- {label} ensemble: {B} members ---")
                S, F, resids = U.fit_members(kind, pr, ctx, B=B)
                _cache_save(cpath, S, F, resids, s_single, p_single, s_resid)

            # ---- single model: deterministic reference + residual intervals ----
            p_single_m = p_single[mk]
            singles_ftu[label] = p_single_m
            rows.append(_point_row(h, f"{label} single", y, p_single_m))
            store[(h, f"{label} single")] = {"p": p_single_m, "lo": None, "hi": None}

            D = U.draw_with_resid(s_single, s_resid, ctx, U.N_SINGLE_DRAWS)[:, mk]
            rows.append(_band_row(h, f"{label}+R", y, D, B=1))
            _, _, lo, hi = U.interval_stats(y, D)
            store[(h, f"{label}+R")] = {"p": D.mean(0), "lo": lo, "hi": hi}
            del D

            # ---- resampled ensemble: parameter uncertainty ----
            Fm = F[:, mk]
            rows.append(_band_row(h, f"{label} ens", y, Fm, B=B))
            fam[label] = Fm

            # ---- + process error ----
            print(f"  + process error: {U.N_NOISE_DRAWS} draws per member")
            A = U.draw_with_resid(S, resids, ctx, U.N_NOISE_DRAWS)[:, mk]
            rows.append(_band_row(h, f"{label}+PE", y, A, B=B))
            _, _, lo, hi = U.interval_stats(y, A)
            store[(h, f"{label}+PE")] = {"p": A.mean(0), "lo": lo, "hi": hi}
            fam_pe[label] = A

        # ---- pooled across families: structural uncertainty ----
        if pool_families and len(fam) > 1:
            POOL = np.concatenate(list(fam.values()), axis=0)
            rows.append(_band_row(h, "POOL ens", y, POOL, B=B * len(fam)))
            del POOL
            head_band = np.concatenate(list(fam_pe.values()), axis=0)
            head_name = "POOL+PE"
            rows.append(_band_row(h, head_name, y, head_band, B=B * len(fam)))
            _, _, lo, hi = U.interval_stats(y, head_band)
            store[(h, head_name)] = {"p": head_band.mean(0), "lo": lo, "hi": hi}

            bud = U.variance_budget(fam, y)
            bud.insert(0, "horizon", h)
            budgets.append(bud)
            print("\nUNCERTAINTY BUDGET (FTU^2)")
            print(bud.round(3).to_string(index=False))
        else:
            k = list(fam_pe)[0]
            head_band, head_name = fam_pe[k], f"{k}+PE"

        if MAKE_SERIES_PER_HORIZON:
            U.fig_series(ts, y, head_band,
                         os.path.join(out_dir, f"fig_h_series_h{h}_{run_name}.png"),
                         f"{run_name}  h={h}", head_name,
                         singles=singles_ftu, persist=pers)

        del fam, fam_pe, head_band

    # =========================================================================
    # TABLES
    # =========================================================================
    tab = pd.DataFrame(rows)
    pbase = tab[tab["model"] == "persistence"].set_index("horizon")["RMSE"]
    tab["SS_vs_persistence"] = tab.apply(
        lambda r: 1 - (r["RMSE"] / pbase[r["horizon"]]) ** 2, axis=1)
    tab = tab[["horizon", "model", "kind", "B", "RMSE", "MAE", "SS_vs_persistence",
               "CRPS_fair", "CRPS_biased", "coverage90", "width90_FTU"]]
    tab = tab.sort_values(["horizon", "RMSE"]).reset_index(drop=True)
    tab.to_csv(os.path.join(out_dir, f"multi_horizon_uq_{run_name}.csv"), index=False)

    print("\n" + "=" * 100)
    print(f"MULTI-HORIZON ENSEMBLE UQ   [{run_name}]")
    print("=" * 100)
    print(tab.round(4).to_string(index=False))

    budget = pd.DataFrame()
    if budgets:
        budget = pd.concat(budgets, ignore_index=True)
        budget.to_csv(os.path.join(
            out_dir, f"multi_horizon_budget_{run_name}.csv"), index=False)
        print("\nUNCERTAINTY BUDGET BY HORIZON")
        print(budget[["horizon", "MSE_total", "pct_parameter",
                      "pct_structural", "pct_unexplained"]].round(2)
              .to_string(index=False))

    best = (tab[tab["model"] != "persistence"]
            .sort_values("RMSE").groupby("horizon").head(1)
            .set_index("horizon")["model"])
    print("\nLOWEST RMSE PER HORIZON")
    for h in horizons:
        print(f"  h={h}: {best[h]}")

    # =========================================================================
    # FIGURES
    # =========================================================================
    fig_horizon_skill(tab, os.path.join(out_dir, f"fig_h_skill_{run_name}.png"),
                      run_name)

    hs = scatter_horizon or horizons[-1]
    name = best[hs]
    rec = store.get((hs, name))
    if rec is None:                                   # e.g. "RF ens", not stored
        name = f"{name.split()[0]}+PE"
        rec = store[(hs, name)]
    band = rec
    if band["lo"] is None:                            # point forecast: borrow a band
        alt = (hs, f"{name.split()[0]}+R")
        band = store.get(alt, {"lo": None, "hi": None})
        if band["lo"] is not None:
            print(f"\n  {name} is a point forecast; 90% band taken from {alt[1]}")
    print(f"  scatter: {name} at h = {hs}")
    fig_scatter(ctxs[hs]["y_te_ftu"][masks[hs]], rec["p"],
                band["lo"], band["hi"], ctxs[hs]["ts_te"][masks[hs]],
                name, hs,
                os.path.join(out_dir, f"fig_h_scatter_{run_name}.png"), run_name)

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min -> {out_dir}")
    return {"table": tab, "budget": budget, "store": store,
            "ctxs": ctxs, "masks": masks, "run": run_name}