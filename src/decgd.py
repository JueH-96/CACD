"""DECGD public interface (Step 3).

Thin object-oriented layer over ``decgd_core`` (formal Modules 1-3 from Step 1):

* ``ErgonomicModel`` subclasses (Module 2) with the benchmark-wide playable ranges:
  guitar 24-fret standard tuning (MIDI 40-88), 5-string 24-fret bass (MIDI 23-67),
  88-key keyboard (MIDI 21-108).
* ``GenerativePriorModel``: the interface every prior must implement, i.e. batched
  log P(S_t | H_t) for candidate onset sets given a stream history. Implemented by
  ``generative_prior.NeuralPolyphonicPrior`` (Step 3) and ``ToyGaussianPrior`` below.

Data types
  Note    = (pitch:int, duration_s:float)
  Cluster = (onset_s:float, tuple[Note, ...])        one onset cluster of ONE stream
  History = list[Cluster]                            chronological; may contain empty sets
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

import decgd_core as core

Note = tuple
Cluster = tuple


# =============================================================== Module 2 classes
class ErgonomicModel:
    """Phi_ergo(S | S_prev) for one instrument family (wraps decgd_core.phi_ergo)."""
    inst: str = ""
    pitch_range: tuple[int, int] = (0, 127)

    def static(self, pitches: Sequence[int]) -> float:
        return core.phi_ergo_static(tuple(sorted(int(p) for p in pitches)), self.inst)[0]

    def __call__(self, pitches: Sequence[int], prev_pitches: Sequence[int] = ()) -> float:
        return core.phi_ergo(tuple(sorted(int(p) for p in pitches)),
                             tuple(sorted(int(p) for p in prev_pitches)), self.inst)

    def playable_single(self, pitch: int) -> bool:
        return bool(np.isfinite(self.static([pitch])))


class GuitarErgonomicModel(ErgonomicModel):
    inst = "guitar"
    tuning = core.GUITAR_STRINGS
    max_fret = core.GUITAR_MAX_FRET
    pitch_range = (tuning[0], tuning[-1] + max_fret)          # 40-88


class BassErgonomicModel(ErgonomicModel):
    inst = "bass"
    tuning = core.BASS_STRINGS
    max_fret = core.BASS_MAX_FRET
    pitch_range = (tuning[0], tuning[-1] + max_fret)          # 23-67


class KeyboardErgonomicModel(ErgonomicModel):
    inst = "keyboard"
    pitch_range = core.KEYBOARD_RANGE                         # 21-108


ERGO = {"guitar": GuitarErgonomicModel(), "bass": BassErgonomicModel(), "keyboard": KeyboardErgonomicModel()}


# =============================================================== prior interface
class GenerativePriorModel(ABC):
    """Interface: log P_theta(S_t | H_t) for a batch of candidate onset sets."""

    @abstractmethod
    def logp_candidates(self, family: str, history: list[Cluster], onset_s: float,
                        candidates: Sequence[tuple[Note, ...]]) -> np.ndarray:
        """Return log P(S | H, onset_s) for each candidate set S (float array)."""

    def logp(self, family, history, onset_s, S) -> float:
        return float(self.logp_candidates(family, history, onset_s, [S])[0])


class ToyGaussianPrior(GenerativePriorModel):
    """Step-1 toy prior (decgd_core.log_p_set) behind the common interface."""

    def logp_candidates(self, family, history, onset_s, candidates):
        h = core.history_stats([tuple(p for p, _ in notes) for _, notes in history])
        return np.array([core.log_p_set(tuple(p for p, _ in S), h) for S in candidates])


def swap_score(lp11, lp12, lp22, lp21, n1, n2, eps=core.EPS, alpha=1.0):
    """Module 1 on pre-computed log-probabilities: lpXY = log P(S_X | H_Y)."""
    return (lp11 - lp12) / (n1 + eps) ** alpha + (lp22 - lp21) / (n2 + eps) ** alpha


# =============================================================== full DECGD decoder (Step 4, revised)
from dataclasses import asdict, dataclass  # noqa: E402


def _log1mexp(x):
    """log(1 - exp(x)) for x < 0 (numerically stable)."""
    x = np.minimum(x, -1e-7)
    return np.where(x > -0.693, np.log(-np.expm1(x)), np.log1p(-np.exp(x)))


@dataclass
class DECGDConfig:
    # Module 1: activity-conditioned symmetric swap contrast + density equilibrium
    alpha: float = 0.0          # denominator (|S|+eps)^alpha (non-empty streams only)
    eps: float = 1.0
    w_swap: float = 0.25
    w_lm: float = 0.0           # length-normalised activity-conditioned fit (uncontrasted)
    w_empty: float = 0.5        # penalty per empty stream at an onset
    w_dens: float = 0.0         # running density-equilibrium penalty |N0-N1|/(N0+N1)
    # Module 2: soft ergonomics + voice leading; infeasible sets get a per-stream penalty
    w_ergo: float = 1.0
    w_jump: float = 1.0
    w_reg: float = 1.0
    w_cross: float = 1.0
    ema: float = 0.3
    infeasible_pen: float = 1e6
    beam: int = 4
    # Module 3: suffix/window block-swap refinement + acoustic/pitch prototype anchoring
    refine: bool = False
    lookahead: int = 2
    lambda_ac: float = 0.0
    ac_feat: str = "raw+pitch"  # raw+pitch | resid | raw
    margin_q: float = 0.2
    max_win: int = 6
    sweeps: int = 2


def _labelings_for(notes_p, symmetric):
    """All labelings, or (identical histories) pitch-ordered splits lower->0; the split at the largest
    pitch gap is listed first so that ties are broken deterministically and musically."""
    from baselines import labelings
    m = len(notes_p)
    if not symmetric:
        return labelings(m)
    order = np.argsort(notes_p, kind="stable")
    ps = np.asarray(notes_p)[order]
    ks = list(range(m, -1, -1))
    if m >= 2:
        kgap = int(np.argmax(np.diff(ps))) + 1
        ks = [kgap] + [k for k in ks if k != kgap]
    out = []
    for k in ks:
        lab = np.ones(m, int)
        lab[order[:k]] = 0
        out.append(lab)
    return out


class DECGD:
    """Dual-Stream Equilibrium Contrastive Generative Decoding on real note lists.

    Revisions after the Step-4 dev failure (review):
      * per-stream histories contain ONLY the stream's non-empty onsets (no [TS, END] rest steps),
        with per-stream inter-onset times;
      * Module 1 uses activity-conditioned log P(S | S != {}, H) and contributes 0 for an empty stream;
      * ergonomic infeasibility is a symmetric per-stream penalty (no early return);
      * Module 3 performs true suffix swaps (flip j..T-1) and window swaps, scoring LM terms over the
        transition window plus the full-excerpt prototype log-likelihood.
    """

    def __init__(self, prior, cfg: DECGDConfig, lp_cache: dict | None = None):
        self.prior, self.cfg = prior, cfg
        self._lp_cache = {} if lp_cache is None else lp_cache   # (ctx tuple, subset) -> raw log P

    def _logp_pairs_cached(self, pairs):
        keys = [(tuple(c), S) for c, S in pairs]
        miss = list(dict.fromkeys(k for k in keys if k not in self._lp_cache))
        if miss:
            vals = self.prior.logp_pairs([(list(c), S) for c, S in miss])
            self._lp_cache.update(zip(miss, vals.tolist()))
        return np.array([self._lp_cache[k] for k in keys])

    # ---------------------------------------------------------------- per-cluster terms
    def terms(self, fam, S, lpA, lpB, last, reg, cnt):
        """lpA[v]/lpB[v] = activity-conditioned log P(S_v | H_0)/(H_1) (ignored for empty S_v)."""
        c = self.cfg
        n = [len(S[0]), len(S[1])]
        delta = 0.0
        if n[0]:
            delta += (lpA[0] - lpB[0]) / (n[0] + c.eps) ** c.alpha
        if n[1]:
            delta += (lpB[1] - lpA[1]) / (n[1] + c.eps) ** c.alpha
        s = c.w_swap * delta
        if c.w_lm:
            s += c.w_lm * ((lpA[0] / (n[0] + 1) if n[0] else 0.0) + (lpB[1] / (n[1] + 1) if n[1] else 0.0))
        P = [tuple(sorted(p for p, _ in S[v])) for v in (0, 1)]
        n_inf = 0
        for v in (0, 1):
            if not P[v]:
                s -= c.w_empty
                continue
            e = core.phi_ergo(P[v], last[v], fam)
            if np.isfinite(e):
                s += c.w_ergo * e
            else:
                n_inf += 1
            mu = float(np.mean(P[v]))
            if last[v]:
                s -= c.w_jump * abs(mu - np.mean(last[v])) / 12
            if reg[v] is not None:
                s -= c.w_reg * abs(mu - reg[v]) / 12
        if P[0] and P[1] and reg[0] is not None and reg[1] is not None:
            lo, hi = (0, 1) if reg[0] <= reg[1] else (1, 0)
            s -= c.w_cross * float(max(P[lo]) > min(P[hi]))
        if c.w_dens:
            N0, N1 = cnt[0] + n[0], cnt[1] + n[1]
            s -= c.w_dens * abs(N0 - N1) / max(1, N0 + N1)
        return s - c.infeasible_pen * n_inf

    def _update(self, st, S, t):
        from generative_prior import step_tokens
        h, tt = [list(st["hist"][0]), list(st["hist"][1])], list(st["t"])
        last, reg, cnt = list(st["last"]), list(st["reg"]), list(st["cnt"])
        for v in (0, 1):
            if not S[v]:
                continue                            # only non-empty events enter a stream's history
            h[v] += step_tokens(4.0 if tt[v] is None else t - tt[v], S[v])
            tt[v] = t
            P = tuple(sorted(p for p, _ in S[v]))
            mu = float(np.mean(P))
            reg[v] = mu if reg[v] is None else (1 - self.cfg.ema) * reg[v] + self.cfg.ema * mu
            last[v] = P
            cnt[v] += len(P)
        return dict(hist=h, t=tt, last=last, reg=reg, cnt=cnt)

    @staticmethod
    def _init_state():
        return dict(hist=[[], []], t=[None, None], last=[(), ()], reg=[None, None], cnt=[0, 0])

    def _ctx(self, fam, st, v, t):
        return self.prior.context(fam, st["hist"][v], st["t"][v], t)

    def _notes(self, ex, c):
        n = ex["notes"]
        return [(int(n.pitch.values[i]), float(n.duration.values[i])) for i in c]

    @staticmethod
    def _cond(lp_row, lp_empty):
        """Activity-conditioned log-probabilities log P(S | S != {}, H)."""
        return lp_row - _log1mexp(lp_empty)

    # ---------------------------------------------------------------- forward beam
    def forward(self, ex):
        fam, on = ex["family"], ex["notes"].onset.values
        beam = [dict(score=0.0, labs=[], st=self._init_state(), info=[])]
        for c in ex["clusters"]:
            t = float(on[c[0]])
            notes = self._notes(ex, c)
            per_item, ctxs, subsets = [], [], {()}
            for it in beam:
                sym = it["st"]["hist"][0] == it["st"]["hist"][1]
                labs = _labelings_for([p for p, _ in notes], sym)
                Ss = [[tuple(sorted(notes[i] for i in np.where(l == v)[0])) for v in (0, 1)] for l in labs]
                for S in Ss:
                    subsets.update(S)
                ctxs += [self._ctx(fam, it["st"], v, t) for v in (0, 1)]
                per_item.append((labs, Ss))
            subsets = sorted(subsets)
            sidx = {s: k for k, s in enumerate(subsets)}
            lp = self._logp_pairs_cached([(cx, S) for cx in ctxs for S in subsets]).reshape(len(ctxs), len(subsets))
            lp = self._cond(lp, lp[:, [sidx[()]]])
            cand = []
            for k, (it, (labs, Ss)) in enumerate(zip(beam, per_item)):
                A, B = lp[2 * k], lp[2 * k + 1]
                sc = np.array([self.terms(fam, S, [A[sidx[S[0]]], A[sidx[S[1]]]], [B[sidx[S[0]]], B[sidx[S[1]]]],
                                          it["st"]["last"], it["st"]["reg"], it["st"]["cnt"]) for S in Ss])
                order = np.argsort(-sc, kind="stable")
                margin = float(sc[order[0]] - sc[order[1]]) if len(sc) > 1 else np.inf
                tops = [labs[j].tolist() for j in order[:3]]
                for j in order[: self.cfg.beam]:
                    cand.append((it["score"] + sc[j], k, j, margin, tops))
            cand.sort(key=lambda x: -x[0])
            beam = [dict(score=tot, labs=beam[k]["labs"] + [per_item[k][0][j]],
                         st=self._update(beam[k]["st"], per_item[k][1][j], t),
                         info=beam[k]["info"] + [dict(margin=mg, tops=tp)])
                    for tot, k, j, mg, tp in cand[: self.cfg.beam]]
        best = beam[0]
        pred = np.zeros(len(ex["notes"]), int)
        for c, l in zip(ex["clusters"], best["labs"]):
            pred[c] = l
        return pred, [np.asarray(l) for l in best["labs"]], best["info"]

    # ---------------------------------------------------------------- Module 3
    def _prototypes(self, ex, labs, conf_mask):
        from sklearn.decomposition import PCA
        from baselines import pitch_residualise
        n = ex["notes"]
        pitch = n.pitch.values.astype(float)
        F = pitch_residualise(ex["fz"], pitch) if self.cfg.ac_feat == "resid" else ex["fz"]
        F = PCA(n_components=6, random_state=0).fit_transform(F)
        F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        if self.cfg.ac_feat == "raw+pitch":
            F = np.c_[F, (pitch - pitch.mean()) / (pitch.std() + 1e-9)]
        lab_note, conf_note = np.zeros(len(n), int), np.zeros(len(n), bool)
        for c, l, ok in zip(ex["clusters"], labs, conf_mask):
            lab_note[c], conf_note[c] = l, ok
        if min(np.sum(conf_note & (lab_note == v)) for v in (0, 1)) < 8:
            return None
        mu = [F[conf_note & (lab_note == v)].mean(0) for v in (0, 1)]
        sd = [F[conf_note & (lab_note == v)].std(0) + 0.3 for v in (0, 1)]
        return np.stack([(-0.5 * ((F - mu[v]) / sd[v]) ** 2 - np.log(sd[v])).sum(1) for v in (0, 1)], 1)

    def window_score(self, ex, labs, lo, hi, ac_ll):
        """LM + voice-leading terms over [lo, hi+lookahead] (prefix-conditioned histories, i.e. prefix and
        continuation scores) + full-excerpt prototype log-likelihood contrast."""
        fam, on = ex["family"], ex["notes"].onset.values
        T = len(ex["clusters"])
        end = min(T - 1, hi + self.cfg.lookahead)
        Ss, st, states = [], self._init_state(), []
        for t_i, (c, l) in enumerate(zip(ex["clusters"], labs)):
            notes = self._notes(ex, c)
            Ss.append([tuple(sorted(notes[i] for i in np.where(l == v)[0])) for v in (0, 1)])
        for t_i in range(end + 1):
            if t_i >= lo:
                states.append(st)
            st = self._update(st, Ss[t_i], float(on[ex["clusters"][t_i][0]]))
        pairs = []
        for k, t_i in enumerate(range(lo, end + 1)):
            t = float(on[ex["clusters"][t_i][0]])
            ctx = [self._ctx(fam, states[k], v, t) for v in (0, 1)]
            pairs += [(ctx[0], Ss[t_i][0]), (ctx[0], Ss[t_i][1]), (ctx[0], ()),
                      (ctx[1], Ss[t_i][0]), (ctx[1], Ss[t_i][1]), (ctx[1], ())]
        lp = self._logp_pairs_cached(pairs).reshape(-1, 2, 3)
        J = 0.0
        for k, t_i in enumerate(range(lo, end + 1)):
            A = self._cond(lp[k, 0, :2], lp[k, 0, 2])
            B = self._cond(lp[k, 1, :2], lp[k, 1, 2])
            J += self.terms(fam, Ss[t_i], A, B, states[k]["last"], states[k]["reg"], states[k]["cnt"])
        if ac_ll is not None and self.cfg.lambda_ac:
            for c, l in zip(ex["clusters"], labs):
                J += self.cfg.lambda_ac * float(np.sum(ac_ll[c, l] - ac_ll[c, 1 - l]))
        return J

    def refine(self, ex, labs, info):
        c = self.cfg
        T = len(labs)
        margins = np.array([i["margin"] for i in info])
        fin = np.isfinite(margins)
        thr = np.quantile(margins[fin], c.margin_q) if fin.any() else -np.inf
        flag = fin & (margins <= thr)
        P = [[ex["notes"].pitch.values[cl][l == v] for v in (0, 1)] for cl, l in zip(ex["clusters"], labs)]
        a0 = np.concatenate([p[0] for p in P] + [[0]])
        a1 = np.concatenate([p[1] for p in P] + [[0]])
        low = 0 if np.median(a0) <= np.median(a1) else 1
        for t in range(T):
            if len(P[t][low]) and len(P[t][1 - low]) and P[t][low].max() > P[t][1 - low].min():
                flag[t] = True
        ac_ll = self._prototypes(ex, labs, ~flag) if c.lambda_ac else None
        wins = []
        for t in np.where(flag)[0]:
            lo, hi = max(0, t - 1), min(T - 1, t + 1)
            if wins and lo <= wins[-1][1] + 1 and hi - wins[-1][0] < c.max_win:
                wins[-1][1] = max(wins[-1][1], hi)
            else:
                wins.append([lo, hi])
        labs = [np.asarray(l).copy() for l in labs]
        n_moves = 0
        base = {}                                   # (wlo, whi) -> score of the current trajectory
        for lo, hi in wins:
            for _ in range(c.sweeps):
                improved = False
                moves = [("suffix", j) for j in range(lo, hi + 1)] + [("window", j) for j in range(lo, hi + 1)] + \
                        [("alt", t, np.asarray(a)) for t in range(lo, hi + 1) for a in info[t]["tops"][1:]]
                for mv in moves:
                    trial = [l.copy() for l in labs]
                    if mv[0] == "suffix":                       # true suffix swap: flip j..T-1
                        for u in range(mv[1], T):
                            trial[u] = 1 - trial[u]
                        wlo, whi = max(0, mv[1] - 1), mv[1]
                    elif mv[0] == "window":                     # local segment swap j..hi
                        for u in range(mv[1], hi + 1):
                            trial[u] = 1 - trial[u]
                        wlo, whi = max(0, mv[1] - 1), hi
                    else:
                        trial[mv[1]] = mv[2].copy()
                        wlo, whi = mv[1], mv[1]
                    if all(np.array_equal(a, b) for a, b in zip(trial, labs)):
                        continue
                    if (wlo, whi) not in base:
                        base[(wlo, whi)] = self.window_score(ex, labs, wlo, whi, ac_ll)
                    gain = self.window_score(ex, trial, wlo, whi, ac_ll) - base[(wlo, whi)]
                    if gain > 1e-6:
                        labs, improved = trial, True
                        n_moves += 1
                        base = {}
                if not improved:
                    break
        pred = np.zeros(len(ex["notes"]), int)
        for cl, l in zip(ex["clusters"], labs):
            pred[cl] = l
        return pred, dict(n_windows=len(wins), n_flagged=int(flag.sum()), n_moves=n_moves)

    def decode(self, ex, forward_out=None):
        pred, labs, info = forward_out or self.forward(ex)
        if not self.cfg.refine:
            return pred, {}
        return self.refine(ex, labs, info)


REFINE_KEYS = ("refine", "lookahead", "lambda_ac", "ac_feat", "margin_q", "max_win", "sweeps")


def forward_key(cfg: dict) -> tuple:
    """Configs with equal forward_key share the same forward pass (refinement settings excluded)."""
    full = asdict(DECGDConfig(**cfg))
    return tuple(sorted((k, v) for k, v in full.items() if k not in REFINE_KEYS))


def decode_many(prior, ex, cfgs: dict) -> dict:
    """Decode one excerpt under several configs, computing each distinct forward pass once and
    sharing one (context, subset) log-prob cache across all configs."""
    cache, fwd, out = {}, {}, {}
    for name, cfg in cfgs.items():
        dec = DECGD(prior, DECGDConfig(**cfg), lp_cache=cache)
        fk = forward_key(cfg)
        if fk not in fwd:
            fwd[fk] = dec.forward(ex)
        out[name] = dec.decode(ex, forward_out=fwd[fk])
    return out


def config_dict(cfg: DECGDConfig) -> dict:
    return asdict(cfg)
