"""DECGD v2 (the hybrid / Step 7): hybrid generative-discriminative two-stream decoding.

Extends decgd.DECGD (the classic code, unchanged) with:
  * Discriminative B3 term (out-of-fold B3 proposals; see b3_v2_proposals.py), two parts:
      node  : sum_i log P_B3(stream of i), with B3's arbitrary global labelling aligned ONLINE to each beam
              trajectory by a decayed agreement score A_t = d*A_{t-1} + sum_i (2 l_i - 1) logit(p_i)
              (prefix/window alignment; while |A| < align_thr the node term is the max over both alignments,
              so it is permutation symmetric at the start);
      pair  : permutation-invariant pairwise same-stream log-likelihood from B3's pair logits Z_ij, for every
              note i of C_t vs the trajectory's notes j in the previous `pair_win` seconds (mean over j) and
              vs the other notes of C_t (mean over partners).
  * Proposals: per beam item, all labelings of C_t (2^m for m <= 10, else 2(m+1) pitch splits) are ranked by the
    B3 score and the top `n_prop` (plus the 2 the classic pitch-split labelings) are scored by the full objective,
    including the activity-conditioned swap-contrastive prior term (the classic Module 1).
  * Equilibrium: w_eq * 2|N_1/N - target_1|, target_1 = B3's aligned mean stream-1 probability (0.5 without B3),
    i.e. density balance that respects B3's voice-activity estimate.
  * Refinement (Module 3+): the classic suffix swaps, window swaps and alternative candidates, plus single-note flips in
    clusters with >= 3 notes, with clusters flagged by low beam margin, register crossing, or high-confidence
    (|p - .5| > b3_flag_conf) disagreement with B3 under GLOBAL consensus alignment. Moves are scored over the
    window plus lookahead (continuation) with the full hybrid objective; the node term is scored over the whole
    excerpt with global alignment (max over both labellings -> exactly permutation invariant).
Without B3 inputs (w_node = w_pair = 0) and w_eq = w_dens the forward pass reduces to the classic DECGD.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SESSION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import decgd as D  # noqa: E402
import decgd_core as core  # noqa: E402

PCLIP = 0.02
ZCLIP = 6.0


# ======================================================================= data
def load_dev():
    """Dev excerpts only (manifest split == 'dev'); file reads are logged by the data-access guard if installed."""
    import baselines as BL
    man = json.loads((SESSION / "results" / "dataset_manifest.json").read_text())
    out = []
    for e in man["excerpts"]:
        if e["split"] != "dev":
            continue
        pn, pf = SESSION / e["files"]["notes.csv"]["path"], SESSION / e["files"]["features_z.npy"]["path"]
        assert "/data/dev/" in str(pn) and "/data/dev/" in str(pf)
        n = pd.read_csv(pn)
        out.append(dict(id=e["excerpt_id"], family=e["family"], source=e["source"], group=e["piece_group"], notes=n,
                        fz=np.load(pf).astype(np.float64), clusters=BL.cluster_index(n.onset.values),
                        stratum=("GuitarSet guitar" if e["source"] == "guitarset" else f"Lakh {e['family']}")))
    return out


def load_folds(EX):
    f = json.loads((SESSION / "results" / "step3_cache" / "folds.json").read_text())
    return np.array([int(f[e["id"]]) for e in EX])


def attach_b3(ex, p, pair):
    """Attach B3 outputs (node prob of label 1, pair logits) to an excerpt dict (copy)."""
    p = np.clip(np.asarray(p, float), PCLIP, 1 - PCLIP)
    Z = np.clip(np.asarray(pair, float), -ZCLIP, ZCLIP)
    ls = -np.logaddexp(0, -Z)                      # log sigmoid(Z): same stream
    ld = -np.logaddexp(0, Z)                       # log sigmoid(-Z): different streams
    on = ex["notes"].onset.values
    return {**ex, "b3": dict(lp1=np.log(p), lp0=np.log(1 - p), logit=np.log(p) - np.log(1 - p), ls=ls, ld=ld,
                             pmean=float(p.mean()), on=on)}


# ======================================================================= config
@dataclass
class DECGDv2Config(D.DECGDConfig):
    w_node: float = 0.0
    w_pair: float = 0.0
    w_eq: float = 0.0
    n_prop: int = 6
    pair_win: float = 2.0
    align_decay: float = 0.95
    align_thr: float = 1.0
    b3_flag_conf: float = 0.3
    flip_moves: bool = True
    eq_target: str = "b3"      # "b3": B3 aligned activity ratio; "balanced": 0.5 (Step-8 equilibrium-strengthened hybrids)


REFINE_KEYS_V2 = D.REFINE_KEYS + ("b3_flag_conf", "flip_moves")


def forward_key(cfg: dict) -> tuple:
    full = asdict(DECGDv2Config(**cfg))
    return tuple(sorted((k, v) for k, v in full.items() if k not in REFINE_KEYS_V2))


# ======================================================================= decoder
class DECGDv2(D.DECGD):
    def __init__(self, prior, cfg: DECGDv2Config, lp_cache=None):
        super().__init__(prior, cfg, lp_cache)
        assert cfg.w_dens == 0.0, "use w_eq (B3-aware equilibrium) in v2"

    @property
    def use_lm(self):
        return bool(self.cfg.w_swap or self.cfg.w_lm)

    @property
    def use_b3(self):
        return bool(self.cfg.w_node or self.cfg.w_pair)

    # ------------------------------------------------------------ B3 terms
    def _pair_terms(self, ex, c, Lm, nl):
        """Pair score for candidate labelings Lm (K, m) of cluster c given note labels nl (-1 = unassigned)."""
        b = ex["b3"]
        on = b["on"]
        t = on[c[0]]
        lo = int(np.searchsorted(on, t - self.cfg.pair_win - 1e-9))
        pidx = np.arange(lo, c[0])
        pidx = pidx[nl[pidx] >= 0]
        m = len(c)
        val = np.zeros((2, m))
        if len(pidx):
            S, Dd = b["ls"][np.ix_(c, pidx)], b["ld"][np.ix_(c, pidx)]
            lab = nl[pidx][None, :]
            for a in (0, 1):
                val[a] = np.where(lab == a, S, Dd).mean(1)
        hist = val[Lm, np.arange(m)[None, :]].sum(1)
        within = np.zeros(len(Lm))
        if m > 1:
            S, Dd = b["ls"][np.ix_(c, c)], b["ld"][np.ix_(c, c)]
            iu = np.triu_indices(m, 1)
            same = Lm[:, iu[0]] == Lm[:, iu[1]]
            within = np.where(same, S[iu][None], Dd[iu][None]).sum(1) / (m - 1)
        return hist + within

    def _node_terms(self, ex, c, Lm, A):
        b = ex["b3"]
        f0 = np.where(Lm == 1, b["lp1"][c][None], b["lp0"][c][None]).sum(1)
        f1 = np.where(Lm == 1, b["lp0"][c][None], b["lp1"][c][None]).sum(1)
        if abs(A) < self.cfg.align_thr:
            return np.maximum(f0, f1)
        return f0 if A > 0 else f1

    def _eq_term(self, ex, cnt_after, A):
        c = self.cfg
        if not c.w_eq:
            return 0.0
        N = cnt_after[0] + cnt_after[1]
        if N == 0:
            return 0.0
        fr1 = cnt_after[1] / N
        if self.use_b3 and "b3" in ex and c.eq_target == "b3":
            tg = ex["b3"]["pmean"]
            if abs(A) < c.align_thr:
                return -c.w_eq * 2 * min(abs(fr1 - tg), abs(fr1 - (1 - tg)))
            tg = tg if A > 0 else 1 - tg
        else:
            tg = 0.5
        return -c.w_eq * 2 * abs(fr1 - tg)

    # ------------------------------------------------------------ candidates
    def _candidates(self, ex, c, notes, it, sym):
        from baselines import labelings
        pv = [p for p, _ in notes]
        base = D._labelings_for(pv, sym)
        if not self.use_b3:
            return base, np.zeros(len(base))
        m = len(c)
        allL = labelings(m)
        if sym:                                    # empty identical histories: lowest note -> stream 0
            lo = int(np.argmin(pv))
            allL = [l for l in allL if l[lo] == 0]
        Lm = np.array(allL)
        ds = self.cfg.w_node * self._node_terms(ex, c, Lm, it["A"]) + self.cfg.w_pair * self._pair_terms(ex, c, Lm, it["nl"])
        keep = list(np.argsort(-ds, kind="stable")[: self.cfg.n_prop])
        cand, seen = [], set()
        for j in keep:
            cand.append(Lm[j])
            seen.add(tuple(Lm[j]))
        for l in base[:2]:
            if tuple(l) not in seen:
                cand.append(np.asarray(l))
                seen.add(tuple(l))
        Lc = np.array(cand)
        ds_c = self.cfg.w_node * self._node_terms(ex, c, Lc, it["A"]) + self.cfg.w_pair * self._pair_terms(ex, c, Lc, it["nl"])
        return [np.asarray(l) for l in cand], ds_c

    # ------------------------------------------------------------ forward
    def forward(self, ex):
        fam, on = ex["family"], ex["notes"].onset.values
        N = len(on)
        beam = [dict(score=0.0, labs=[], st=self._init_state(), info=[], nl=-np.ones(N, int), A=0.0)]
        for c in ex["clusters"]:
            t = float(on[c[0]])
            notes = self._notes(ex, c)
            per_item, ctxs, subsets = [], [], {()}
            for it in beam:
                sym = it["st"]["hist"][0] == it["st"]["hist"][1]
                labs, ds = self._candidates(ex, c, notes, it, sym)
                Ss = [[tuple(sorted(notes[i] for i in np.where(l == v)[0])) for v in (0, 1)] for l in labs]
                if self.use_lm:                       # prior unused (w_swap = w_lm = 0; prior may be None)
                    for S in Ss:
                        subsets.update(S)
                    ctxs += [self._ctx(fam, it["st"], v, t) for v in (0, 1)]
                per_item.append((labs, Ss, ds))
            if self.use_lm:
                subsets = sorted(subsets)
                sidx = {s: k for k, s in enumerate(subsets)}
                lp = self._logp_pairs_cached([(cx, S) for cx in ctxs for S in subsets]).reshape(len(ctxs), len(subsets))
                lp = self._cond(lp, lp[:, [sidx[()]]])
            cand = []
            for k, (it, (labs, Ss, ds)) in enumerate(zip(beam, per_item)):
                sc = []
                for j, S in enumerate(Ss):
                    if self.use_lm:
                        A_, B_ = lp[2 * k], lp[2 * k + 1]
                        la, lb = [A_[sidx[S[0]]], A_[sidx[S[1]]]], [B_[sidx[S[0]]], B_[sidx[S[1]]]]
                    else:
                        la = lb = [0.0, 0.0]
                    s = self.terms(fam, S, la, lb, it["st"]["last"], it["st"]["reg"], it["st"]["cnt"])
                    cnt = [it["st"]["cnt"][0] + len(S[0]), it["st"]["cnt"][1] + len(S[1])]
                    s += ds[j] + self._eq_term(ex, cnt, it["A"])
                    sc.append(s)
                sc = np.array(sc)
                order = np.argsort(-sc, kind="stable")
                margin = float(sc[order[0]] - sc[order[1]]) if len(sc) > 1 else np.inf
                tops = [labs[j].tolist() for j in order[:3]]
                for j in order[: self.cfg.beam]:
                    cand.append((it["score"] + sc[j], k, j, margin, tops))
            cand.sort(key=lambda x: -x[0])
            new = []
            for tot, k, j, mg, tp in cand[: self.cfg.beam]:
                it, l = beam[k], per_item[k][0][j]
                nl = it["nl"].copy()
                nl[c] = l
                A = it["A"]
                if "b3" in ex:
                    A = self.cfg.align_decay * A + float(np.sum((2 * l - 1) * ex["b3"]["logit"][c]))
                new.append(dict(score=tot, labs=it["labs"] + [l], st=self._update(it["st"], per_item[k][1][j], t),
                                info=it["info"] + [dict(margin=mg, tops=tp)], nl=nl, A=A))
            beam = new
        best = beam[0]
        pred = np.zeros(N, int)
        for c, l in zip(ex["clusters"], best["labs"]):
            pred[c] = l
        return pred, [np.asarray(l) for l in best["labs"]], best["info"]

    # ------------------------------------------------------------ refinement
    def _global_node(self, ex, nl):
        b = ex["b3"]
        f0 = np.where(nl == 1, b["lp1"], b["lp0"]).sum()
        f1 = np.where(nl == 1, b["lp0"], b["lp1"]).sum()
        return max(f0, f1), (1 if f0 >= f1 else -1)

    def window_score(self, ex, labs, lo, hi, ac_ll):
        fam, on = ex["family"], ex["notes"].onset.values
        T = len(ex["clusters"])
        end = min(T - 1, hi + self.cfg.lookahead)
        nl = np.zeros(len(on), int)
        Ss = []
        for c, l in zip(ex["clusters"], labs):
            nl[c] = l
            notes = self._notes(ex, c)
            Ss.append([tuple(sorted(notes[i] for i in np.where(l == v)[0])) for v in (0, 1)])
        st, states = self._init_state(), []
        for t_i in range(end + 1):
            if t_i >= lo:
                states.append(st)
            st = self._update(st, Ss[t_i], float(on[ex["clusters"][t_i][0]]))
        if self.use_lm:
            pairs = []
            for k, t_i in enumerate(range(lo, end + 1)):
                t = float(on[ex["clusters"][t_i][0]])
                ctx = [self._ctx(fam, states[k], v, t) for v in (0, 1)]
                pairs += [(ctx[0], Ss[t_i][0]), (ctx[0], Ss[t_i][1]), (ctx[0], ()),
                          (ctx[1], Ss[t_i][0]), (ctx[1], Ss[t_i][1]), (ctx[1], ())]
            lp = self._logp_pairs_cached(pairs).reshape(-1, 2, 3)
        J = 0.0
        has_b3 = "b3" in ex and self.use_b3
        gn, sgn = self._global_node(ex, nl) if "b3" in ex else (0.0, 1)
        A_glob = 1e9 * sgn                          # global consensus alignment for the equilibrium target
        for k, t_i in enumerate(range(lo, end + 1)):
            if self.use_lm:
                A_ = self._cond(lp[k, 0, :2], lp[k, 0, 2])
                B_ = self._cond(lp[k, 1, :2], lp[k, 1, 2])
            else:
                A_ = B_ = [0.0, 0.0]
            J += self.terms(fam, Ss[t_i], A_, B_, states[k]["last"], states[k]["reg"], states[k]["cnt"])
            cnt = [states[k]["cnt"][0] + len(Ss[t_i][0]), states[k]["cnt"][1] + len(Ss[t_i][1])]
            J += self._eq_term(ex, cnt, A_glob if "b3" in ex else 0.0)
            if has_b3 and self.cfg.w_pair:
                c = ex["clusters"][t_i]
                nl_prev = nl.copy()
                nl_prev[c[0]:] = -1
                J += self.cfg.w_pair * float(self._pair_terms(ex, c, nl[c][None, :], nl_prev)[0])
        if has_b3 and self.cfg.w_node:
            J += self.cfg.w_node * gn
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
        n_b3_flags = 0
        if "b3" in ex and self.use_b3:
            nl = np.zeros(len(ex["notes"]), int)
            for cl, l in zip(ex["clusters"], labs):
                nl[cl] = l
            _, sgn = self._global_node(ex, nl)
            p1 = np.exp(ex["b3"]["lp1"]) if sgn > 0 else np.exp(ex["b3"]["lp0"])
            for t, cl in enumerate(ex["clusters"]):
                conf = np.abs(p1[cl] - 0.5) > c.b3_flag_conf
                if np.any(conf & ((p1[cl] > 0.5).astype(int) != labs[t])):
                    if not flag[t]:
                        n_b3_flags += 1
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
        n_moves, base = 0, {}
        for lo, hi in wins:
            for _ in range(c.sweeps):
                improved = False
                moves = [("suffix", j) for j in range(lo, hi + 1)] + [("window", j) for j in range(lo, hi + 1)] + \
                        [("alt", t, np.asarray(a)) for t in range(lo, hi + 1) for a in info[t]["tops"][1:]]
                if c.flip_moves:
                    moves += [("flip", t, i) for t in range(lo, hi + 1) if len(labs[t]) >= 3 for i in range(len(labs[t]))]
                for mv in moves:
                    trial = [l.copy() for l in labs]
                    if mv[0] == "suffix":
                        for u in range(mv[1], T):
                            trial[u] = 1 - trial[u]
                        wlo, whi = max(0, mv[1] - 1), mv[1]
                    elif mv[0] == "window":
                        for u in range(mv[1], hi + 1):
                            trial[u] = 1 - trial[u]
                        wlo, whi = max(0, mv[1] - 1), hi
                    elif mv[0] == "alt":
                        trial[mv[1]] = mv[2].copy()
                        wlo, whi = mv[1], mv[1]
                    else:
                        trial[mv[1]][mv[2]] = 1 - trial[mv[1]][mv[2]]
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
        return pred, dict(n_windows=len(wins), n_flagged=int(flag.sum()), n_b3_flags=n_b3_flags, n_moves=n_moves)


def decode_many(prior, ex, cfgs: dict, cache=None) -> dict:
    """Decode one excerpt under several v2 configs, sharing forward passes and the LM cache."""
    cache = {} if cache is None else cache
    fwd, out = {}, {}
    for name, cfg in cfgs.items():
        dec = DECGDv2(prior, DECGDv2Config(**cfg), lp_cache=cache)
        fk = forward_key(cfg)
        if fk not in fwd:
            fwd[fk] = dec.forward(ex)
        out[name] = dec.decode(ex, forward_out=fwd[fk])
    return out
