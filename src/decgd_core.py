"""DECGD formal core: toy generative prior, Modules 1-3 and a greedy decoder.

Notation
--------
Onset cluster t carries a set of simultaneous notes C_t. A two-stream
assignment splits C_t = S_t^(1) U S_t^(2) (either part may be empty).
H_<t^(v) is the history of stream v (its last onset sets, incl. empty sets).

Generative prior P_theta(S | H)
    Autoregressive within-onset factorisation with a canonical (ascending
    pitch) order and an explicit step-termination event:

    log P(S | H) = sum_{i=0}^{k-1} [ log P(cont_i | H) + log q_H(p_i) ] + log P(stop_k | H)

    P(cont_0)=1-p0_H, P(stop_0)=p0_H, P(cont_i>=1)=rho_H, P(stop_k>=1)=1-rho_H,
    q_H(p) ∝ pi(p) * N(p; mu_H, sigma_H)   (pi = history-independent marginal).

    The model has k+1 events for a k-note set, hence the per-event normaliser
    |S| + eps with eps = 1 (see Module 1).

Module 1 - Symmetric Swap-Contrastive Generative Prior
    Delta_swap = [log P(S1|H1) - log P(S1|H2)] / (|S1|+eps)
               + [log P(S2|H2) - log P(S2|H1)] / (|S2|+eps)
    Any history-independent additive term h(S) of log P (marginal pitch
    frequencies, token costs, a global length prior g(|S|)) cancels exactly.

Module 2 - Polyphonic Ergonomic Span Constraint Phi_ergo(S | S_prev, inst)
    Hard feasibility (-inf) + soft hinge penalties for guitar (6-string,
    standard tuning), bass (4-string) and keyboard (two hands).

Module 3 - Lookahead Bidirectional Refinement with Acoustic Anchoring
    Flags high-entropy / predicted-crossing onsets, merges them into windows and
    runs block-coordinate ascent over label swaps with an objective combining
    forward+lookahead and backward (time-reversed) swap-contrast scores, the
    ergonomic potential and a note-level acoustic swap-contrast against
    stream prototypes estimated from confident onsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, permutations, product

import numpy as np

PITCHES = np.arange(128)
LOG_PI = -0.5 * ((PITCHES - 60.0) / 14.0) ** 2  # broad history-independent marginal
LOG_PI = LOG_PI - np.log(np.exp(LOG_PI).sum())


# =========================================================== generative prior
@dataclass(frozen=True)
class HistStats:
    p0: float      # P(empty onset)
    rho: float     # P(another note | >=1 note already)
    mu: float
    sigma: float


def history_stats(hist: list[tuple[int, ...]], L_onsets: int = 8, L_notes: int = 12) -> HistStats:
    """Smoothed sufficient statistics of a stream history (list of onset sets)."""
    recent = hist[-L_onsets:]
    n_on = len(recent)
    ks = np.array([len(s) for s in recent], float) if n_on else np.zeros(0)
    n_empty = float(np.sum(ks == 0))
    p0 = (n_empty + 1.0) / (n_on + 2.0)
    nonempty = ks[ks > 0]
    extra = float(np.sum(nonempty - 1)) if len(nonempty) else 0.0
    rho = (extra + 0.5) / (extra + len(nonempty) + 1.0)
    notes = [p for s in hist[::-1] for p in s][:L_notes]
    if notes:
        a = np.array(notes, float)
        mu = a.mean()
        sigma = np.sqrt((np.sum((a - mu) ** 2) + 4 * 9.0) / (len(a) + 4))
    else:
        mu, sigma = 60.0, 12.0
    return HistStats(p0, rho, float(mu), float(max(sigma, 1.5)))


@lru_cache(maxsize=200_000)
def _log_q(mu: float, sigma: float) -> np.ndarray:
    lg = LOG_PI - 0.5 * ((PITCHES - mu) / sigma) ** 2
    return lg - np.logaddexp.reduce(lg)


def log_p_set(S: tuple[int, ...], h: HistStats, extra_h=None) -> float:
    """log P_theta(S | H). ``extra_h`` is an optional history-independent
    additive term h(S) used by the invariance test."""
    S = tuple(sorted(S))
    k = len(S)
    lq = _log_q(round(h.mu, 3), round(h.sigma, 3))
    lp = 0.0
    if k == 0:
        lp = np.log(h.p0)
    else:
        lp += np.log(1 - h.p0) + (k - 1) * np.log(h.rho) + np.log(1 - h.rho)
        lp += float(np.sum(lq[list(S)]))
    if extra_h is not None:
        lp += extra_h(S)
    return float(lp)


class CorpusLengthPrior:
    """History-independent length/density term g_beta(|S|) mimicking the corpus
    marginal that a pretrained AR LM regresses to when context is weak:

        g(0) = beta * log pi0,   g(k>=1) = beta * [log(1-pi0) + (k-1) log rho0]

    With pi0=0.8 (most per-stream steps are rests) and rho0=0.7 (chords are
    long), merging two non-empty sets into one saves
    beta*[log pi0 - log(1-pi0) + log rho0] ~ 1.03*beta nats per onset under
    naive joint scoring -> density collapse. Picklable for multiprocessing."""

    def __init__(self, beta: float, pi0: float = 0.8, rho0: float = 0.7):
        self.beta, self.pi0, self.rho0 = beta, pi0, rho0

    def __call__(self, S) -> float:
        k = len(S)
        if k == 0:
            return self.beta * np.log(self.pi0)
        return self.beta * (np.log(1 - self.pi0) + (k - 1) * np.log(self.rho0))


# ================================================================ Module 1
EPS = 1.0


def score_naive(S1, S2, h1, h2, extra_h=None):
    return log_p_set(S1, h1, extra_h) + log_p_set(S2, h2, extra_h)


def score_naive_lennorm(S1, S2, h1, h2, extra_h=None):
    return (log_p_set(S1, h1, extra_h) / (len(S1) + EPS)
            + log_p_set(S2, h2, extra_h) / (len(S2) + EPS))


def score_swap(S1, S2, h1, h2, extra_h=None, eps: float = EPS, normalise: bool = True,
               alpha: float = 1.0):
    """Module 1: symmetric swap-contrastive generative prior Delta_swap.

    Denominator (|S|+eps)^alpha: alpha=0 recovers the unnormalised swap score,
    alpha=1 full per-event normalisation (Step-1 default). alpha is a tunable
    hyperparameter to be selected on the dev split only (review item, Step 2)."""
    t1 = log_p_set(S1, h1, extra_h) - log_p_set(S1, h2, extra_h)
    t2 = log_p_set(S2, h2, extra_h) - log_p_set(S2, h1, extra_h)
    if normalise:
        return t1 / (len(S1) + eps) ** alpha + t2 / (len(S2) + eps) ** alpha
    return t1 + t2


# ================================================================ Module 2
# Step 3: widened to the benchmark's playable ranges (Step 1 used 20 frets / 4-string bass)
GUITAR_STRINGS = (40, 45, 50, 55, 59, 64)   # E2 A2 D3 G3 B3 E4, 24 frets -> MIDI 40-88
BASS_STRINGS = (23, 28, 33, 38, 43)         # B0 E1 A1 D2 G2 (5-string), 24 frets -> MIDI 23-67
GUITAR_MAX_FRET = 24
BASS_MAX_FRET = 24
KEYBOARD_RANGE = (21, 108)


def _fretboard_best(S, strings, max_fret, hard_span, soft_span):
    """Min-penalty injective string assignment. Returns (span, position) of the
    best fingering, or None if infeasible."""
    S = sorted(S)
    if len(S) > len(strings):
        return None
    best = None
    for strs in permutations(range(len(strings)), len(S)):
        frets = [p - strings[s] for p, s in zip(S, strs)]
        if any(f < 0 or f > max_fret for f in frets):
            continue
        fretted = [f for f in frets if f > 0]
        span = (max(fretted) - min(fretted)) if fretted else 0
        if span > hard_span:
            continue
        pos = min(fretted) if fretted else 0
        key = (max(0, span - soft_span), span, pos)
        if best is None or key < best[0]:
            best = (key, span, pos)
    return None if best is None else (best[1], best[2])


@lru_cache(maxsize=100_000)
def phi_ergo_static(S: tuple[int, ...], inst: str) -> tuple[float, float]:
    """(penalty, hand_position) of the simultaneous set S; penalty=-inf if
    physically infeasible. Hand position is used for the transition term."""
    S = tuple(sorted(S))
    if len(S) == 0:
        return 0.0, np.nan
    if inst == "guitar":
        r = _fretboard_best(S, GUITAR_STRINGS, GUITAR_MAX_FRET, hard_span=5, soft_span=3)
        if r is None:
            return -np.inf, np.nan
        span, pos = r
        return -1.0 * max(0, span - 3), float(pos)
    if inst == "bass":
        if len(S) > 3:
            return -np.inf, np.nan
        r = _fretboard_best(S, BASS_STRINGS, BASS_MAX_FRET, hard_span=5, soft_span=3)
        if r is None:
            return -np.inf, np.nan
        span, pos = r
        return -1.0 * max(0, span - 3) - 1.0 * (len(S) - 1), float(pos)
    if inst == "keyboard":
        if len(S) > 10 or S[0] < KEYBOARD_RANGE[0] or S[-1] > KEYBOARD_RANGE[1]:
            return -np.inf, np.nan
        best = -np.inf
        for cut in range(0, len(S) + 1):   # left hand = S[:cut], right = S[cut:]
            pen = 0.0
            ok = True
            for hand in (S[:cut], S[cut:]):
                if not hand:
                    continue
                span = hand[-1] - hand[0]
                if len(hand) > 5 or span > 14:
                    ok = False
                    break
                pen -= 0.5 * max(0, span - 12)
            if ok:
                best = max(best, pen)
        return best, float(np.mean(S))
    raise ValueError(f"unknown instrument {inst}")


def phi_ergo(S, S_prev, inst: str, jump_free: float = 5.0, jump_w: float = 0.25) -> float:
    """Module 2: Phi_ergo(S_t | S_{t-1}, inst) = static span potential +
    hinge penalty on hand-position jumps between consecutive non-empty onsets
    (frets for guitar/bass, semitones/12*5 for keyboard)."""
    pen, pos = phi_ergo_static(tuple(sorted(S)), inst)
    if not np.isfinite(pen) or not S or not S_prev:
        return pen
    _, ppos = phi_ergo_static(tuple(sorted(S_prev)), inst)
    if np.isnan(pos) or np.isnan(ppos):
        return pen
    jump = abs(pos - ppos) if inst != "keyboard" else abs(pos - ppos) * 5.0 / 12.0
    return pen - jump_w * max(0.0, jump - jump_free)


# ===================================================== decoding utilities
def partitions(notes: tuple[int, ...]):
    """All 2^m labelings of the m notes of an onset cluster."""
    for lab in product((0, 1), repeat=len(notes)):
        S1 = tuple(p for p, l in zip(notes, lab) if l == 0)
        S2 = tuple(p for p, l in zip(notes, lab) if l == 1)
        yield np.array(lab), S1, S2


@dataclass
class DecoderConfig:
    scorer: str = "swap"            # naive | naive_lennorm | swap_unnorm | swap
    use_ergo: bool = False
    inst: tuple[str, str] = ("guitar", "guitar")
    lambda_ergo: float = 1.0
    refine: bool = False
    use_acoustic: bool = False
    lambda_ac: float = 0.5
    entropy_tau: float = 0.5
    lookahead: int = 3
    max_sweeps: int = 4
    temperature: float = 1.0
    extra_h: object = field(default=None, repr=False)


def local_score(S1, S2, h1, h2, prev1, prev2, cfg: DecoderConfig) -> float:
    if cfg.scorer == "naive":
        s = score_naive(S1, S2, h1, h2, cfg.extra_h)
    elif cfg.scorer == "naive_lennorm":
        s = score_naive_lennorm(S1, S2, h1, h2, cfg.extra_h)
    elif cfg.scorer == "swap_unnorm":
        s = score_swap(S1, S2, h1, h2, cfg.extra_h, normalise=False)
    elif cfg.scorer == "swap":
        s = score_swap(S1, S2, h1, h2, cfg.extra_h)
    else:
        raise ValueError(cfg.scorer)
    if cfg.use_ergo:
        e = phi_ergo(S1, prev1, cfg.inst[0]) + phi_ergo(S2, prev2, cfg.inst[1])
        if not np.isfinite(e):
            return -np.inf
        s += cfg.lambda_ergo * e
    return s


def _last_nonempty(hist):
    for s in reversed(hist):
        if s:
            return s
    return ()


def greedy_decode(clusters: list[tuple[int, ...]], prime_labels: list[np.ndarray],
                  cfg: DecoderConfig, acoustic=None):
    """Greedy left-to-right decoding. ``clusters[t]`` is the ascending tuple of
    pitches at onset t; the first len(prime_labels) onsets are given.
    Returns labels per onset and the normalised posterior entropy per onset."""
    hist = ([], [])
    labels, entropy, cand = [], [], []
    for t, notes in enumerate(clusters):
        if t < len(prime_labels):
            lab = prime_labels[t]
            entropy.append(0.0)
            cand.append([lab])
        else:
            h1, h2 = history_stats(hist[0]), history_stats(hist[1])
            p1, p2 = _last_nonempty(hist[0]), _last_nonempty(hist[1])
            labs, scores = [], []
            for lab, S1, S2 in partitions(notes):
                labs.append(lab)
                scores.append(local_score(S1, S2, h1, h2, p1, p2, cfg))
            scores = np.array(scores)
            if not np.isfinite(scores).any():   # nothing feasible: drop ergo
                scores = np.array([local_score(S1, S2, h1, h2, p1, p2,
                                               DecoderConfig(scorer=cfg.scorer, extra_h=cfg.extra_h))
                                   for _, S1, S2 in partitions(notes)])
            order = np.argsort(-scores, kind="stable")
            lab = labs[order[0]]
            fin = np.isfinite(scores)
            z = scores[fin] / cfg.temperature
            pr = np.exp(z - z.max())
            pr /= pr.sum()
            ent = float(-(pr * np.log(pr + 1e-300)).sum() / np.log(max(2, len(scores))))
            entropy.append(ent)
            cand.append([labs[i] for i in order[:4]])
        labels.append(np.asarray(lab))
        hist[0].append(tuple(p for p, l in zip(notes, lab) if l == 0))
        hist[1].append(tuple(p for p, l in zip(notes, lab) if l == 1))
    out = (labels, np.array(entropy), cand)
    if cfg.refine:
        out = refine(clusters, labels, np.array(entropy), cand, len(prime_labels), cfg, acoustic)
    return out


# ================================================================ Module 3
def _streams(clusters, labels):
    s1 = [tuple(p for p, l in zip(c, lab) if l == 0) for c, lab in zip(clusters, labels)]
    s2 = [tuple(p for p, l in zip(c, lab) if l == 1) for c, lab in zip(clusters, labels)]
    return s1, s2


def window_objective(clusters, labels, t_lo, t_hi, cfg, acoustic=None, proto=None):
    """J(window) = forward swap score over [t_lo, t_hi + lookahead]
    + backward (time-reversed) swap score over [t_lo, t_hi]
    + lambda_ergo * Phi_ergo + lambda_ac * acoustic swap-contrast."""
    s1, s2 = _streams(clusters, labels)
    T = len(clusters)
    J = 0.0
    fwd_cfg = DecoderConfig(scorer="swap", use_ergo=cfg.use_ergo, inst=cfg.inst,
                            lambda_ergo=cfg.lambda_ergo, extra_h=cfg.extra_h)
    for t in range(t_lo, min(T, t_hi + 1 + cfg.lookahead)):
        h1, h2 = history_stats(s1[:t]), history_stats(s2[:t])
        J += local_score(s1[t], s2[t], h1, h2, _last_nonempty(s1[:t]), _last_nonempty(s2[:t]), fwd_cfg)
    bwd_cfg = DecoderConfig(scorer="swap", extra_h=cfg.extra_h)
    for t in range(t_lo, t_hi + 1):
        f1 = s1[t + 1:][::-1]
        f2 = s2[t + 1:][::-1]
        h1, h2 = history_stats(f1), history_stats(f2)
        J += local_score(s1[t], s2[t], h1, h2, (), (), bwd_cfg)
    if cfg.use_acoustic and acoustic is not None and proto is not None:
        for t in range(t_lo, t_hi + 1):
            lab = labels[t]
            ll = proto["ll"][t]   # (m, 2) log-lik of each note under proto 0 / 1
            if len(lab):
                own = ll[np.arange(len(lab)), lab]
                oth = ll[np.arange(len(lab)), 1 - lab]
                J += cfg.lambda_ac * float(np.sum(own - oth))
    return J


def acoustic_prototypes(acoustic, labels, confident):
    """Diagonal-Gaussian prototypes per predicted stream from confident onsets;
    returns per-onset (m, 2) log-likelihood arrays."""
    feats = {0: [], 1: []}
    for t, ok in enumerate(confident):
        if ok:
            for f, l in zip(acoustic[t], labels[t]):
                feats[int(l)].append(f)
    if min(len(feats[0]), len(feats[1])) < 5:
        return None
    mus = [np.mean(feats[v], 0) for v in (0, 1)]
    sds = [np.std(feats[v], 0) + 1e-3 for v in (0, 1)]
    ll = []
    for t in range(len(acoustic)):
        a = np.asarray(acoustic[t]).reshape(-1, len(mus[0]))
        ll.append(np.stack([np.sum(-0.5 * ((a - mus[v]) / sds[v]) ** 2 - np.log(sds[v]), 1)
                            for v in (0, 1)], 1))
    return dict(mu=mus, sd=sds, ll=ll)


def predicted_crossing(clusters, labels):
    """Onsets where the predicted low stream's max pitch exceeds the predicted
    high stream's min pitch (labels decide low/high by median pitch)."""
    s1, s2 = _streams(clusters, labels)
    a1 = [p for s in s1 for p in s]
    a2 = [p for s in s2 for p in s]
    if not a1 or not a2:
        return np.zeros(len(clusters), bool)
    low, high = (s1, s2) if np.median(a1) <= np.median(a2) else (s2, s1)
    return np.array([bool(l and h and max(l) > min(h)) for l, h in zip(low, high)])


def refine(clusters, labels, entropy, cand, n_prime, cfg, acoustic=None):
    """Module 3: detect uncertain windows and run block-coordinate ascent."""
    labels = [np.asarray(l).copy() for l in labels]
    T = len(clusters)
    cross = predicted_crossing(clusters, labels)
    flag = (entropy > cfg.entropy_tau) | cross
    flag[:n_prime] = False
    proto = acoustic_prototypes(acoustic, labels, ~flag) if (cfg.use_acoustic and acoustic is not None) else None
    # merge flagged onsets (dilated by 1) into windows
    idx = np.where(flag)[0]
    windows = []
    for t in idx:
        lo, hi = max(n_prime, t - 1), min(T - 1, t + 1)
        if windows and lo <= windows[-1][1] + 1:
            windows[-1][1] = max(windows[-1][1], hi)
        else:
            windows.append([lo, hi])
    n_moves = 0
    for lo, hi in windows:
        best = window_objective(clusters, labels, lo, hi, cfg, acoustic, proto)
        for _ in range(cfg.max_sweeps):
            improved = False
            moves = []
            for t in range(lo, hi + 1):
                for c in cand[t]:
                    if not np.array_equal(c, labels[t]):
                        moves.append(("set", t, c))
                moves.append(("block", t, None))      # swap identities from t to hi
            for kind, t, c in moves:
                trial = [l.copy() for l in labels]
                if kind == "set":
                    trial[t] = np.asarray(c).copy()
                else:
                    for u in range(t, hi + 1):
                        trial[u] = 1 - trial[u]
                J = window_objective(clusters, trial, lo, hi, cfg, acoustic, proto)
                if J > best + 1e-9:
                    best, labels, improved = J, trial, True
                    n_moves += 1
            if not improved:
                break
    return labels, entropy, cand, dict(n_windows=len(windows), n_moves=n_moves,
                                       n_flagged=int(flag.sum()))
