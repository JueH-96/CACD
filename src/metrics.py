"""Evaluation metrics and paired statistics for DECGD stream separation.

All metrics operate on one *piece* at a time. A piece is described by
note-aligned arrays (the assignment setting: the same set of notes is given to
every system, only the stream label differs):

    onsets : float array (seconds or beats), shape (N,)
    pitches: int array (MIDI pitch), shape (N,)
    gt     : int array of ground-truth stream ids in {0..K-1}, shape (N,)
    pred   : int array of predicted stream ids in {0..K-1}; -1 = unassigned

Permutation invariance: stream ids are arbitrary labels (e.g. "guitar A" vs
"guitar B" of the same MIDI program), so every piece-level metric is computed
under the single *global* label permutation that maximises the metric for that
piece. Permutations are NOT re-optimised per onset; local disagreement with the
global permutation is what VSER measures.

Statistics: paired Wilcoxon signed-rank, Holm-Bonferroni, paired percentile
bootstrap CI (10,000 resamples by default), matched-pairs rank-biserial
correlation and Cliff's delta.
"""

from __future__ import annotations

from itertools import permutations
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

__all__ = [
    "onset_clusters", "best_permutation", "pi_accuracy", "pi_macro_f1",
    "voice_swap_error_rate", "crossing_mask", "voice_crossing_accuracy",
    "evaluate_piece", "paired_wilcoxon", "rank_biserial_paired", "cliffs_delta",
    "paired_bootstrap_ci", "holm_correction", "compare_to_baseline",
]


# --------------------------------------------------------------------------- utils
def _as_arrays(gt, pred):
    gt = np.asarray(gt, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if gt.shape != pred.shape:
        raise ValueError(f"gt and pred shape mismatch: {gt.shape} vs {pred.shape}")
    if gt.ndim != 1:
        raise ValueError("gt/pred must be 1-D note-aligned arrays")
    if (gt < 0).any():
        raise ValueError("ground-truth stream ids must be >= 0")
    return gt, pred


def _n_streams(gt, pred, n_streams):
    if n_streams is not None:
        return int(n_streams)
    return int(max(gt.max(initial=0), pred.max(initial=0)) + 1)


def _apply_perm(pred, perm):
    """Relabel predicted ids: pred id j -> perm[j]; -1 stays unassigned."""
    out = pred.copy()
    m = pred >= 0
    out[m] = np.asarray(perm)[pred[m]]
    return out


def onset_clusters(onsets, tol: float = 1e-6) -> np.ndarray:
    """Group notes into onset clusters (notes whose onsets differ by <= tol
    from the cluster start). Returns an int cluster index per note, ordered in
    time."""
    onsets = np.asarray(onsets, dtype=float)
    order = np.argsort(onsets, kind="stable")
    cid = np.empty(len(onsets), dtype=int)
    c, start = -1, -np.inf
    for i in order:
        if onsets[i] - start > tol:
            c += 1
            start = onsets[i]
        cid[i] = c
    return cid


# ------------------------------------------------------------------ PI metrics
def _accuracy(gt, pred):
    return float(np.mean(gt == pred)) if len(gt) else np.nan


def _macro_f1(gt, pred, k):
    f1s = []
    for s in range(k):
        g = gt == s
        p = pred == s
        denom = g.sum() + p.sum()
        if denom == 0:  # stream absent from both: skip (does not reward/penalise)
            continue
        f1s.append(2.0 * np.sum(g & p) / denom)
    return float(np.mean(f1s)) if f1s else np.nan


def best_permutation(gt, pred, n_streams=None, criterion: str = "acc"):
    """Return (perm, score) maximising accuracy ('acc') or macro-F1 ('f1')."""
    gt, pred = _as_arrays(gt, pred)
    k = _n_streams(gt, pred, n_streams)
    fn = _accuracy if criterion == "acc" else (lambda g, p: _macro_f1(g, p, k))
    best = (None, -np.inf)
    for perm in permutations(range(k)):
        sc = fn(gt, _apply_perm(pred, perm))
        if sc > best[1]:
            best = (perm, sc)
    return best


def pi_accuracy(gt, pred, n_streams=None) -> float:
    """Permutation-invariant assignment accuracy (fraction of notes in the
    correct stream under the best global relabelling)."""
    return best_permutation(gt, pred, n_streams, "acc")[1]


def pi_macro_f1(gt, pred, n_streams=None) -> float:
    """Permutation-invariant note-level macro-F1 over streams."""
    return best_permutation(gt, pred, n_streams, "f1")[1]


# ------------------------------------------------------------------- VSER
def voice_swap_error_rate(onsets, gt, pred, n_streams=None, tol: float = 1e-6) -> dict:
    """Voice Swap Error Rate.

    For every onset cluster the *local* best permutation is computed. A cluster
    is *decisive* if the local best permutation is unique (strictly better
    than all others). VSER = number of changes of the local permutation between
    consecutive decisive clusters / (n_decisive - 1); analogous to identity
    switches in multi-object tracking. ``swapped_fraction`` is the fraction of
    decisive clusters whose local permutation differs from the global one.
    """
    gt, pred = _as_arrays(gt, pred)
    k = _n_streams(gt, pred, n_streams)
    cid = onset_clusters(onsets, tol)
    gperm, _ = best_permutation(gt, pred, k, "acc")
    perms = list(permutations(range(k)))
    local = []
    for c in range(cid.max() + 1 if len(cid) else 0):
        m = cid == c
        scores = np.array([np.sum(gt[m] == _apply_perm(pred[m], p)) for p in perms])
        top = scores.max()
        if np.sum(scores == top) == 1:
            local.append(perms[int(scores.argmax())])
    n_dec = len(local)
    switches = sum(1 for a, b in zip(local[:-1], local[1:]) if a != b)
    return dict(
        vser=switches / (n_dec - 1) if n_dec > 1 else np.nan,
        n_switches=int(switches),
        n_decisive_clusters=int(n_dec),
        swapped_fraction=(np.mean([p != gperm for p in local]) if n_dec else np.nan),
    )


# -------------------------------------------------------------- crossing
def crossing_mask(onsets, pitches, gt, window: float = 0.0, tol: float = 1e-6) -> np.ndarray:
    """Boolean mask of notes lying in ground-truth voice-crossing segments.

    The *low* stream is the GT stream with the lower median pitch over the
    piece (2-stream case). For each onset cluster at time t the local registers
    P_t^(low), P_t^(high) are the pitches of each stream with onset in
    [t - window, t + window]. The cluster is a crossing cluster iff both are
    non-empty and max P_t^(low) > min P_t^(high). All notes of a crossing
    cluster are marked.
    """
    onsets = np.asarray(onsets, float)
    pitches = np.asarray(pitches, float)
    gt = np.asarray(gt, int)
    streams = np.unique(gt)
    if len(streams) != 2:
        return np.zeros(len(gt), dtype=bool)
    med = {s: np.median(pitches[gt == s]) for s in streams}
    low, high = sorted(streams, key=lambda s: med[s])
    cid = onset_clusters(onsets, tol)
    mask = np.zeros(len(gt), dtype=bool)
    for c in np.unique(cid):
        m = cid == c
        t = onsets[m][0]
        w = np.abs(onsets - t) <= window + tol
        pl = pitches[w & (gt == low)]
        ph = pitches[w & (gt == high)]
        if len(pl) and len(ph) and pl.max() > ph.min():
            mask[m] = True
    return mask


def voice_crossing_accuracy(onsets, pitches, gt, pred, window: float = 0.0,
                            n_streams=None, tol: float = 1e-6) -> dict:
    """Accuracy on notes inside GT crossing segments, under the piece-level
    global permutation (so a model cannot 'fix' crossings by relabelling them
    locally).

    ``vc_macro_recall`` is the class-balanced companion: the mean over GT streams
    of per-stream recall restricted to crossing notes. A decoder that collapses a
    crossing cluster into the denser stream gets recall 0 for the other stream, so
    collapse cannot inflate this metric (unlike plain ``vc_acc``)."""
    gt, pred = _as_arrays(gt, pred)
    mask = crossing_mask(onsets, pitches, gt, window, tol)
    if not mask.any():
        return dict(vc_acc=np.nan, vc_macro_recall=np.nan, n_crossing_notes=0)
    perm, _ = best_permutation(gt, pred, n_streams, "acc")
    p = _apply_perm(pred, perm)
    rec = [float(np.mean(p[mask & (gt == s)] == s)) for s in np.unique(gt[mask])]
    return dict(vc_acc=float(np.mean(gt[mask] == p[mask])), vc_macro_recall=float(np.mean(rec)),
                n_crossing_notes=int(mask.sum()))


def evaluate_piece(onsets, pitches, gt, pred, crossing_window: float = 0.0, tol: float = 1e-6) -> dict:
    """Compute all piece-level metrics in one call (``tol`` = onset-cluster tolerance;
    use ~0.03 s for real performance data)."""
    out = dict(pi_acc=pi_accuracy(gt, pred), pi_macro_f1=pi_macro_f1(gt, pred))
    out.update(voice_swap_error_rate(onsets, gt, pred, tol=tol))
    out.update(voice_crossing_accuracy(onsets, pitches, gt, pred, crossing_window, tol=tol))
    return out


# ------------------------------------------------------------- statistics
def rank_biserial_paired(a, b) -> float:
    """Matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-),
    zero differences dropped (consistent with zero_method='wilcox')."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def cliffs_delta(a, b) -> float:
    """Cliff's delta = P(A > B) - P(A < B) over all cross pairs (treats the two
    samples as unpaired; reported alongside the paired rank-biserial)."""
    a = np.asarray(a, float)
    b = np.sort(np.asarray(b, float))
    gt = np.searchsorted(b, a, side="left").sum()          # # b < a
    lt = (len(b) - np.searchsorted(b, a, side="right")).sum()  # # b > a
    return float((gt - lt) / (len(a) * len(b)))


def paired_wilcoxon(a, b, alternative: str = "two-sided") -> dict:
    """Two-sided Wilcoxon signed-rank test on paired scores a vs b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal length")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    n_nz = int(np.sum(d != 0))
    if n_nz == 0:
        stat, p = 0.0, 1.0
    else:
        res = stats.wilcoxon(a, b, zero_method="wilcox", alternative=alternative)
        stat, p = float(res.statistic), float(res.pvalue)
    return dict(W=stat, p=p, n_pairs=int(len(d)), n_nonzero=n_nz,
                rank_biserial=rank_biserial_paired(a, b), cliffs_delta=cliffs_delta(a, b),
                median_diff=float(np.median(d)) if len(d) else np.nan,
                mean_diff=float(np.mean(d)) if len(d) else np.nan)


def paired_bootstrap_ci(a, b, n_resamples: int = 10_000, ci: float = 0.95,
                        statistic: Callable[[np.ndarray], np.ndarray] | None = None,
                        seed: int = 42) -> dict:
    """Percentile bootstrap CI of a statistic (default: mean) of the paired
    differences a - b, resampling pairs with replacement."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return dict(estimate=np.nan, ci_low=np.nan, ci_high=np.nan, n_resamples=n_resamples)
    statistic = statistic or (lambda x: x.mean(axis=-1))
    rng = np.random.default_rng(seed)
    boots = []
    chunk = max(1, min(n_resamples, 2_000_000 // max(len(d), 1)))
    done = 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        idx = rng.integers(0, len(d), size=(m, len(d)))
        boots.append(statistic(d[idx]))
        done += m
    boots = np.concatenate(boots)
    alpha = (1 - ci) / 2
    return dict(estimate=float(statistic(d[None, :])[0]),
                ci_low=float(np.quantile(boots, alpha)),
                ci_high=float(np.quantile(boots, 1 - alpha)),
                n_resamples=int(n_resamples))


def holm_correction(pvals: Sequence[float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down adjustment (statsmodels implementation)."""
    p = np.asarray(pvals, float)
    reject, p_adj, _, _ = multipletests(p, alpha=alpha, method="holm")
    return dict(p_adj=p_adj, reject=reject)


def compare_to_baseline(scores: dict[str, Sequence[float]], baseline: str,
                        metric_name: str = "metric", n_resamples: int = 10_000,
                        seed: int = 42) -> pd.DataFrame:
    """Paired comparison of every system against ``baseline`` on per-piece
    scores; Holm correction across the family of comparisons."""
    if baseline not in scores:
        raise KeyError(f"baseline '{baseline}' not in {list(scores)}")
    rows = []
    for name, s in scores.items():
        if name == baseline:
            continue
        w = paired_wilcoxon(s, scores[baseline])
        bs = paired_bootstrap_ci(s, scores[baseline], n_resamples=n_resamples, seed=seed)
        rows.append(dict(metric=metric_name, system=name, baseline=baseline,
                         mean_system=float(np.nanmean(s)), mean_baseline=float(np.nanmean(scores[baseline])),
                         **w, boot_mean_diff=bs["estimate"], boot_ci_low=bs["ci_low"],
                         boot_ci_high=bs["ci_high"]))
    df = pd.DataFrame(rows)
    if len(df):
        h = holm_correction(df["p"].values)
        df["p_holm"] = h["p_adj"]
        df["reject_holm_0.05"] = h["reject"]
    return df
