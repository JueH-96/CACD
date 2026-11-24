"""Shared utilities: method list, paired statistics at excerpt and piece-group level."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import metrics as M

BASELINES = ["B1_Viterbi_DP", "B2_Acoustic_Symbolic_HMM", "B3_Supervised_GNN_Transformer", "B4_Naive_LM_Prior"]
DECGD_VARIANTS = ["DECGD_M1_swap_only", "DECGD_M12_forward", "DECGD_M123_noacoustic", "DECGD_full"]
PRIMARY = "pi_macro_f1"
SECONDARY = ["pi_acc", "swap_rate", "density_imbalance", "vc_macro_recall_w0", "vc_macro_recall_w005"]
HIGHER_BETTER = {"pi_macro_f1": True, "pi_acc": True, "swap_rate": False, "density_imbalance": False,
                 "vc_macro_recall_w0": True, "vc_macro_recall_w005": True, "vc_acc_w0": True, "collapse_rate": False}
SEED = 2026
NBOOT = 10_000


def grouped_bootstrap(diff: np.ndarray, groups: np.ndarray, n=NBOOT, seed=SEED):
    """Percentile CI of the mean excerpt-level difference, resampling piece groups."""
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    sums = np.array([diff[idx[g]].sum() for g in ug])
    cnts = np.array([len(idx[g]) for g in ug])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ug), size=(n, len(ug)))
    boot = sums[pick].sum(1) / cnts[pick].sum(1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def paired(a: np.ndarray, b: np.ndarray, groups: np.ndarray) -> dict:
    """a - b at excerpt level (N) and piece-group level (G = per-group means)."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, groups = a[ok], b[ok], groups[ok]
    d = a - b
    w = M.paired_wilcoxon(a, b)
    ga = pd.Series(a).groupby(groups).mean()
    gb = pd.Series(b).groupby(groups).mean()
    wg = M.paired_wilcoxon(ga.values, gb.values)
    sd = d.std(ddof=1)
    return dict(n_excerpts=int(len(d)), n_groups=int(len(ga)), mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                ci95_grouped_bootstrap=grouped_bootstrap(d, groups),
                ci95_excerpt_bootstrap=[M.paired_bootstrap_ci(a, b, n_resamples=NBOOT, seed=SEED)[k] for k in ("ci_low", "ci_high")],
                wilcoxon_W=w["W"], p_excerpt=w["p"], n_nonzero=w["n_nonzero"], rank_biserial=w["rank_biserial"],
                cohens_dz=float(d.mean() / sd) if sd > 0 else 0.0,
                p_group=wg["p"], rank_biserial_group=wg["rank_biserial"], mean_diff_group=float((ga - gb).mean()))


def describe(x: np.ndarray, groups: np.ndarray) -> dict:
    ok = np.isfinite(x)
    x, g = x[ok], groups[ok]
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), n_groups=int(len(np.unique(g))), mean=float(x.mean()), median=float(np.median(x)),
                iqr=[float(np.percentile(x, 25)), float(np.percentile(x, 75))],
                ci95_grouped_bootstrap=grouped_bootstrap(x, g))


def comparison_family(df: pd.DataFrame, target: str, others: list[str], metric: str) -> list[dict]:
    """target vs each of `others` on `metric`; Holm across the family (both excerpt- and group-level p)."""
    piv = df.pivot(index="excerpt_id", columns="method", values=metric)
    grp = df.drop_duplicates("excerpt_id").set_index("excerpt_id").loc[piv.index, "piece_group"].values
    rows = []
    for o in others:
        r = paired(piv[target].values, piv[o].values, grp)
        rows.append(dict(metric=metric, a=target, b=o, **r))
    for key in ("p_excerpt", "p_group"):
        h = M.holm_correction([r[key] for r in rows])
        for r, pa in zip(rows, h["p_adj"]):
            r[key + "_holm"] = float(pa)
    return rows
