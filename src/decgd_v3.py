"""DECGD v3 (the calibrated / Steps 1-2): capacity-debiased B3 + B2 spectral-acoustic posterior + generative swap prior.

Builds on decgd_v2.DECGDv2 (unchanged) and adds
  1. Unsupervised per-excerpt capacity estimate rho_hat in (0, 0.5] (target minor-stream note share), the
     equal-weight mean of three label-free estimates (pre-declared, not tuned):
       rho_clust : onset-cluster capacity model - every multi-note cluster (m >= 2) gives exactly one note to the
                   minor stream (dyads split 1:1, chord+melody clusters split (m-1):1), solo notes (m = 1) split 1:1
                   -> (#clusters with m >= 2 + 0.5 #solo notes) / N;
       rho_b2    : minor share of the B2 spectral-HMM hard labels;
       rho_b3    : min(p_bar, 1 - p_bar) of the B3 node probabilities.
  2. Capacity-aware B3 logit debiasing (1-D optimal transport of the logit marginal):
       (a) chunk polarity alignment: the excerpt is cut into `chunk_s`-second chunks; a chunk's node logits are
           sign-flipped if that increases agreement with B3's own pairwise co-assignment logits Z_ij to the
           already-aligned earlier notes (|dt| <= 2 s) by more than `chunk_margin`;
       (b) quantile centring: z <- z - eta * q, q = the (1 - f1)-quantile of z, f1 = rho_hat if the B3 mean
           probability is < .5 else 1 - rho_hat, so the hard marginal matches the capacity target instead of a
           naive 0 cut (eta = 1 full debiasing, eta = 0 raw logits);
       (c) within-cluster standardisation: for clusters with m >= 2, z_i += w_rel * (z_i - mean_c z) (shift
           invariant, keeps dense chord splits sharp under a global logit shift).
  3. B2 posterior: soft per-note stream log-posteriors from the B2 acoustic-symbolic HMM (class-conditional
     Gaussian log-likelihoods of the converged B2 model, log-softmax with temperature `b2_temp`), polarity-aligned
     to the debiased B3 labels.  B2's pitch-continuity transition ("stick") is represented by DECGD's
     voice-leading terms (w_jump / w_reg), which are the same functional form.
  4. Fused node potential lp_v(i) = gamma_b3 * log sigma(+-z_i) + gamma_b2 * log softmax_B2(i)[v]; pair term
     gamma_b3 * w_pair_rel * B3 pair log-likelihood (permutation invariant); capacity equilibrium
     w_cap * 2|N_1/N - target_1| with target_1 = rho_hat (aligned to the fused polarity; replaces the v2 fixed
     0.28 imbalance cap and the B3 mean-probability target); generative swap-contrastive prior (w_swap) and
     ergonomic / voice-leading terms (w_ergo, w_jump, w_reg, w_cross) from DECGD v1.
  5. Decoding: v2 beam forward pass on the fused objective, plus warm-start proposals (debiased-B3 hard labels,
     B2 hard labels, fused hard labels); every proposal is scored by the full objective J(y) over the whole
     excerpt; the best is refined by block-coordinate ascent over clusters (all labelings for m <= 4, top fused
     labelings + single-note flips otherwise, plus suffix swaps) on clusters where the proposals disagree or the
     fused node margin is small; the final trajectory maximises J.
All terms are invariant to a global relabelling 0 <-> 1 of the output (tested in test_decgd_v3_dev.py).
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SESSION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import decgd_v2 as V  # noqa: E402

PCLIP, ZCLIP = V.PCLIP, V.ZCLIP
LOGIT_CLIP = 8.0


# ======================================================================= B2 soft posterior
def b2_soft(ex, w_ac, w_p, stick, resid, n_iter=3, beam=4, seed=0):
    """baselines.b2_decode, additionally returning the converged class-conditional log-likelihoods ll (N, 2)."""
    from sklearn.cluster import SpectralClustering
    from sklearn.decomposition import PCA
    import baselines as BL
    n = ex["notes"]
    pitch = n.pitch.values.astype(float)
    on = n.onset.values
    A = BL.pitch_residualise(ex["fz"], pitch) if resid else ex["fz"]
    A = PCA(n_components=6, random_state=seed).fit_transform(A)
    A = (A - A.mean(0)) / (A.std(0) + 1e-9)
    X = np.c_[w_ac * A, w_p * (pitch - pitch.mean()) / 6.0]
    D2 = ((X[:, None] - X[None]) ** 2).sum(-1) / X.shape[1]
    W = np.exp(-D2) * (np.abs(on[:, None] - on[None]) <= 2.0)
    np.fill_diagonal(W, 1.0)
    lab = SpectralClustering(2, affinity="precomputed", random_state=seed, assign_labels="discretize").fit_predict(W + 1e-6)

    def gauss_ll(lab_):
        mu = [X[lab_ == v].mean(0) for v in (0, 1)]
        sd = [X[lab_ == v].std(0) + 0.3 for v in (0, 1)]
        return np.stack([(-0.5 * ((X - mu[v]) / sd[v]) ** 2 - np.log(sd[v])).sum(1) for v in (0, 1)], 1)

    for _ in range(n_iter):
        if len(set(lab)) < 2:
            break
        ll = gauss_ll(lab)
        Bm = [(0.0, [], [None, None])]
        for c in ex["clusters"]:
            new = []
            for cost, labs, last in Bm:
                for l in BL.labelings(len(c)):
                    s = ll[c, l].sum()
                    nl = list(last)
                    for v in (0, 1):
                        sel = pitch[c][l == v]
                        if len(sel):
                            if last[v] is not None:
                                s -= stick * abs(sel.mean() - last[v]) / 12
                            nl[v] = sel.mean()
                    new.append((cost + s, labs + [l], nl))
            new.sort(key=lambda x: -x[0])
            Bm = new[:beam]
        lab = np.zeros(len(n), int)
        for c, l in zip(ex["clusters"], Bm[0][1]):
            lab[c] = l
    ll = gauss_ll(lab) if len(set(lab)) == 2 else np.zeros((len(n), 2))
    return lab.astype(int), ll


# ======================================================================= capacity estimate
def capacity_estimate(ex, p_b3, b2_lab):
    """Unsupervised minor-stream share rho_hat in (0, 0.5] (no ground-truth labels are read)."""
    N = len(ex["notes"])
    m = np.array([len(c) for c in ex["clusters"]])
    rho_clust = (float((m >= 2).sum()) + 0.5 * float(m[m == 1].sum())) / N
    rho_clust = min(rho_clust, 1 - rho_clust)
    f = float(np.mean(b2_lab))
    rho_b2 = min(f, 1 - f)
    pb = float(np.mean(p_b3))
    rho_b3 = min(pb, 1 - pb)
    rho = float(np.clip((rho_clust + rho_b2 + rho_b3) / 3, 0.05, 0.5))
    return dict(rho_hat=rho, rho_clust=rho_clust, rho_b2=rho_b2, rho_b3=rho_b3, target_imbalance=abs(1 - 2 * rho))


# ======================================================================= B3 debiasing
def chunk_align(z, Z, on, chunk_s=8.0, margin=0.0, win=2.0):
    """Sign-flip chunk logits that disagree with B3's own pairwise structure w.r.t. earlier aligned notes."""
    z = z.copy()
    if len(z) == 0:
        return z, 0
    cid = np.floor((on - on[0]) / chunk_s).astype(int)
    S = np.tanh(np.clip(Z, -ZCLIP, ZCLIP) / 2)           # signed same-stream affinity
    near = np.abs(on[:, None] - on[None]) <= win
    n_flip = 0
    for k in np.unique(cid)[1:]:
        I = np.where(cid == k)[0]
        J = np.where(cid < k)[0]
        M = near[np.ix_(I, J)]
        if not M.any():
            continue
        yi, yj = np.sign(z[I])[:, None], np.sign(z[J])[None, :]
        agree = (S[np.ix_(I, J)] * yi * yj * M).sum()     # keep: + agreement; flip: - agreement
        if -agree > agree + margin:
            z[I] = -z[I]
            n_flip += 1
    return z, n_flip


def debias_b3(ex, p, Z, rho, eta=1.0, w_rel=0.5, do_chunk=True, chunk_s=8.0):
    """Capacity-aware debiasing of B3 node logits. Returns debiased logits and diagnostics."""
    p = np.clip(np.asarray(p, float), PCLIP, 1 - PCLIP)
    z = np.log(p) - np.log(1 - p)
    on = ex["notes"].onset.values
    n_flip = 0
    if do_chunk:
        z, n_flip = chunk_align(z, np.asarray(Z, float), on, chunk_s=chunk_s)
    pm = float(np.mean(1 / (1 + np.exp(-z))))
    f1 = rho if pm < 0.5 else 1 - rho                     # target share of label 1 (polarity of B3's majority)
    q = float(np.quantile(z, 1 - f1))
    zc = z - eta * q
    if w_rel:
        for c in ex["clusters"]:
            if len(c) >= 2:
                zc[c] = zc[c] + w_rel * (zc[c] - zc[c].mean())
    zc = np.clip(zc, -LOGIT_CLIP, LOGIT_CLIP)
    return zc, dict(n_chunk_flips=n_flip, shift=float(eta * q), raw_frac1=float((z > 0).mean()),
                    cal_frac1=float((zc > 0).mean()), target_frac1=float(f1))


# ======================================================================= fused excerpt inputs
@dataclass
class V3Inputs:
    gamma_b3: float = 1.0
    gamma_b2: float = 0.5
    w_pair_rel: float = 1.0
    eta: float = 1.0             # 1 = capacity debiasing, 0 = raw B3 logits
    w_rel: float = 0.5
    do_chunk: bool = True
    b2_temp: float = 4.0
    cap_target: str = "rho"      # "rho": unsupervised capacity target; "balanced": 0.5 (v1 density equilibrium);
    #                              "b3raw": v2 B3 mean-probability target


def attach_v3(ex, b3, b2, inp: V3Inputs):
    """b3 = dict(p, pair); b2 = dict(lab, ll). Returns excerpt copy with ex['b3'] holding FUSED node log-potentials
    (lp1/lp0/logit), B3 pair log-likelihoods (ls/ld, used with weight w_pair) and the capacity target ('pmean')."""
    cap = capacity_estimate(ex, b3["p"], b2["lab"])
    rho = cap["rho_hat"]
    zc, dinfo = debias_b3(ex, b3["p"], b3["pair"], rho, eta=inp.eta, w_rel=inp.w_rel if inp.eta else 0.0,
                          do_chunk=inp.do_chunk and bool(inp.eta))
    ls_b3_1, ls_b3_0 = -np.logaddexp(0, -zc), -np.logaddexp(0, zc)
    ll = np.asarray(b2["ll"], float) / inp.b2_temp
    lsm = ll - np.logaddexp(ll[:, :1], ll[:, 1:])           # log softmax over the two B2 classes
    lab2 = np.asarray(b2["lab"], int)
    ref = (zc > 0).astype(int) if inp.gamma_b3 else lab2
    flip2 = int(np.mean(lab2 == ref) < 0.5)                  # align B2 polarity to debiased B3
    b2_1, b2_0 = (lsm[:, 0], lsm[:, 1]) if flip2 else (lsm[:, 1], lsm[:, 0])
    lp1 = inp.gamma_b3 * ls_b3_1 + inp.gamma_b2 * b2_1
    lp0 = inp.gamma_b3 * ls_b3_0 + inp.gamma_b2 * b2_0
    Zc = np.clip(np.asarray(b3["pair"], float), -ZCLIP, ZCLIP)
    q1 = 1 / (1 + np.exp(-(lp1 - lp0)))
    if inp.cap_target == "rho":
        tg1 = rho if q1.mean() < 0.5 else 1 - rho
    elif inp.cap_target == "balanced":
        tg1 = 0.5
    else:
        pr = np.clip(np.asarray(b3["p"], float), PCLIP, 1 - PCLIP)
        tg1 = float(pr.mean())
    hard = dict(b3=(zc > 0).astype(int), b2=lab2 if not flip2 else 1 - lab2, fused=(lp1 > lp0).astype(int))
    return {**ex, "b3": dict(lp1=lp1, lp0=lp0, logit=lp1 - lp0, ls=-np.logaddexp(0, -Zc), ld=-np.logaddexp(0, Zc),
                             pmean=float(tg1), on=ex["notes"].onset.values),
            "v3": dict(cap=cap, debias=dinfo, b2_flipped=flip2, hard=hard, zc=zc)}


# ======================================================================= decoder
@dataclass
class DECGDv3Config(V.DECGDv2Config):
    multi_proposal: bool = True
    bc_sweeps: int = 2
    bc_margin: float = 1.0       # clusters whose min |fused logit| < bc_margin are refined
    bc_max_enum: int = 4
    v2_refine: bool = False      # iteration 2: apply v1/v2 Module-3 refinement (suffix/window swaps, acoustic-pitch
    #                              prototype anchoring lambda_ac) to the beam trajectory before the proposal pool


class DECGDv3(V.DECGDv2):
    """v2 forward pass on the fused potentials + multi-proposal warm start + block-coordinate refinement."""

    def full_score(self, ex, labs):
        return self.window_score(ex, labs, 0, len(labs) - 1, None)

    @staticmethod
    def _to_labs(ex, pred):
        return [np.asarray(pred)[c].astype(int).copy() for c in ex["clusters"]]

    def _alts(self, ex, c, cur):
        from baselines import labelings
        m = len(c)
        if m <= self.cfg.bc_max_enum:
            return [np.asarray(l) for l in labelings(m)]
        b = ex["b3"]
        out = [1 - cur]
        for i in range(m):
            t = cur.copy()
            t[i] = 1 - t[i]
            out.append(t)
        out.append((b["lp1"][c] > b["lp0"][c]).astype(int))
        return out

    def block_refine(self, ex, labs, flags, n_check=2):
        """Block-coordinate ascent on the GLOBAL objective J: per flagged cluster, trial moves are ranked by their
        local (window + lookahead) gain and the best `n_check` with positive local gain are accepted only if they
        increase the full-excerpt J (local gains ignore downstream capacity / register effects)."""
        T = len(labs)
        labs = [l.copy() for l in labs]
        J = self.full_score(ex, labs)
        n_moves = 0
        for _ in range(self.cfg.bc_sweeps):
            improved = False
            for t in np.where(flags)[0]:
                c = ex["clusters"][t]
                base = {}
                cands = []
                trials = [("alt", a) for a in self._alts(ex, c, labs[t]) if not np.array_equal(a, labs[t])]
                trials.append(("suffix", None))
                for kind, a in trials:
                    tr = [l.copy() for l in labs]
                    if kind == "alt":
                        tr[t] = a.copy()
                        lo = t
                    else:
                        for u in range(t, T):
                            tr[u] = 1 - tr[u]
                        lo = max(0, t - 1)
                    if lo not in base:
                        base[lo] = self.window_score(ex, labs, lo, t, None)
                    g = self.window_score(ex, tr, lo, t, None) - base[lo]
                    if g > 1e-6:
                        cands.append((g, tr))
                cands.sort(key=lambda x: -x[0])
                for _, tr in cands[:n_check]:
                    Jt = self.full_score(ex, tr)
                    if Jt > J + 1e-6:
                        labs, J, improved = tr, Jt, True
                        n_moves += 1
                        break
            if not improved:
                break
        return labs, n_moves

    def decode(self, ex, forward_out=None):
        pred_f, labs_f, info = forward_out if forward_out is not None else self.forward(ex)
        if self.cfg.v2_refine:
            pred_f, _ = V.DECGDv2.refine(self, ex, labs_f, info)
            labs_f = self._to_labs(ex, pred_f)
        if not self.cfg.multi_proposal:
            return pred_f, dict(chosen="beam")
        pool = {"beam": labs_f}
        if "v3" in ex:
            for k, h in ex["v3"]["hard"].items():
                pool[k] = self._to_labs(ex, h)
        scores = {k: self.full_score(ex, l) for k, l in pool.items()}
        chosen = max(scores, key=lambda k: (scores[k], k == "beam"))
        labs = pool[chosen]
        # flag clusters where proposals disagree (up to global relabelling) or the fused margin is small
        P = {k: np.concatenate([l for l in v]) for k, v in pool.items()}
        ref = P[chosen]
        dis = np.zeros(len(ref), bool)
        for k, v in P.items():
            vv = v if np.mean(v == ref) >= 0.5 else 1 - v
            dis |= vv != ref
        order = np.concatenate(ex["clusters"])
        dis_note = np.zeros(len(ref), bool)
        dis_note[order] = dis
        mg = np.abs(ex["b3"]["logit"]) < self.cfg.bc_margin
        flags = np.array([bool(dis_note[c].any() or mg[c].any()) for c in ex["clusters"]])
        n_moves = 0
        if self.cfg.refine and flags.any():
            labs, n_moves = self.block_refine(ex, labs, flags)
        pred = np.zeros(len(ex["notes"]), int)
        for c, l in zip(ex["clusters"], labs):
            pred[c] = l
        return pred, dict(chosen=chosen, proposal_scores=scores, n_flagged=int(flags.sum()), n_moves=n_moves,
                          final_score=float(self.full_score(ex, labs)))


REFINE_KEYS_V3 = V.REFINE_KEYS_V2 + ("multi_proposal", "bc_sweeps", "bc_margin", "bc_max_enum", "v2_refine")


def forward_key(cfg: dict) -> tuple:
    full = asdict(DECGDv3Config(**cfg))
    return tuple(sorted((k, v) for k, v in full.items() if k not in REFINE_KEYS_V3))


def decode_many(prior, ex, cfgs: dict, cache=None) -> dict:
    """Decode one (already attach_v3-ed) excerpt under several decoder configs sharing forward passes."""
    cache = {} if cache is None else cache
    fwd, out = {}, {}
    for name, cfg in cfgs.items():
        dec = DECGDv3(prior, DECGDv3Config(**cfg), lp_cache=cache)
        fk = forward_key(cfg)
        if fk not in fwd:
            fwd[fk] = dec.forward(ex)
        out[name] = dec.decode(ex, forward_out=fwd[fk])
    return out


# ======================================================================= test-time B3 ensemble (unseen pieces)
def b3_ensemble_raw(item, device, ckpts):
    """Permutation-aligned average of B3 node probabilities and mean pair logits over several nets (the 5 dev
    fold nets + b3_full_dev_v2.pt at locked test time). Not applied to any confirmation split in this step."""
    import torch
    import baselines as BL
    b = BL.b3_collate([item], device)
    ps, zs = [], []
    for ck in ckpts:
        net = BL.make_b3().to(device)
        sd = torch.load(ck, map_location=device)
        net.load_state_dict(sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd)
        net.eval()
        with torch.no_grad():
            node, pair = net(b["pitch"], b["sym"], b["ac"], b["onset"], b["cid"], b["mask"])
        p = torch.sigmoid(node)[0].cpu().numpy().astype(np.float64)
        if ps and np.mean(np.abs(p - ps[0])) > np.mean(np.abs((1 - p) - ps[0])):
            p = 1 - p
        ps.append(p)
        zs.append(pair[0].cpu().numpy().astype(np.float64))
    return dict(p=np.mean(ps, 0), pair=np.mean(zs, 0))
