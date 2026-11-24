"""Step 3 baselines B1-B4 for two-stream same-timbre note assignment.

B1_Viterbi_DP              beam-Viterbi over onset clusters: pitch continuity, register persistence,
                           Module-2 ergonomic cost, crossing and inactivity penalties.
B2_Acoustic_Symbolic_HMM   per-excerpt spectral clustering on (pitch-residualised acoustic + pitch)
                           affinities, then EM / sequential beam smoothing with Gaussian emissions.
B3_Supervised_GNN_Transformer  note Transformer with same-cluster / temporal attention biases,
                           permutation-invariant BCE + pairwise same-stream (graph) loss.
B4_Naive_LM_Prior          greedy decoding maximising log P(S1|H1) + log P(S2|H2) of the pretrained
                           neural prior (generative_prior.py) with ergonomic pruning only.

Data hygiene: ``load_split`` only serves split == "dev"; every file read is logged to
logs/step3_data_access.log.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

SESSION = Path(__file__).resolve().parents[1]
ACCESS_LOG = SESSION / "logs" / "step3_data_access.log"
TOL = 0.03
import decgd_core as core  # noqa: E402
import metrics as M  # noqa: E402


# ===================================================================== data
def load_split(split="dev"):
    if split != "dev":
        raise PermissionError("Step 3 may only read the dev split (confirmation is locked)")
    man = json.loads((SESSION / "results" / "dataset_manifest.json").read_text())
    out = []
    with open(ACCESS_LOG, "a") as logf:
        for e in man["excerpts"]:
            if e["split"] != "dev":
                continue
            paths = {k: SESSION / e["files"][k]["path"] for k in ("notes.csv", "features_z.npy")}
            for p in paths.values():
                assert "/dev/" in str(p)
                logf.write(str(p.relative_to(SESSION)) + "\n")
            n = pd.read_csv(paths["notes.csv"])
            out.append(dict(id=e["excerpt_id"], family=e["family"], source=e["source"],
                            group=e["piece_group"], notes=n, fz=np.load(paths["features_z.npy"]).astype(np.float64),
                            clusters=cluster_index(n.onset.values)))
    return out


def cluster_index(onsets, tol=TOL):
    """List of note-index arrays per onset cluster (notes are onset-sorted)."""
    cid = M.onset_clusters(onsets, tol)
    return [np.where(cid == c)[0] for c in range(cid.max() + 1)]


def labelings(m):
    """All 2^m labelings for m <= 10, else the 2(m+1) pitch-threshold splits (notes pitch-sorted)."""
    if m <= 10:
        return [np.array(l) for l in product((0, 1), repeat=m)]
    out = []
    for k in range(m + 1):
        a = np.r_[np.zeros(k, int), np.ones(m - k, int)]
        out += [a, 1 - a]
    return out


# ===================================================================== metrics
def evaluate(ex, pred):
    n = ex["notes"]
    on, p, g = n.onset.values, n.pitch.values, n.stream.values
    pred = np.asarray(pred, int)
    r = M.evaluate_piece(on, p, g, pred, crossing_window=0.0, tol=TOL)
    r = dict(pi_acc=r["pi_acc"], pi_macro_f1=r["pi_macro_f1"], swap_rate=r["vser"],
             vc_acc_w0=r["vc_acc"], vc_macro_recall_w0=r["vc_macro_recall"], n_cross_w0=r["n_crossing_notes"])
    for tag, win in (("w005", 0.05), ("w025", 0.25)):
        w = M.voice_crossing_accuracy(on, p, g, pred, window=win, tol=TOL)
        r.update({f"vc_acc_{tag}": w["vc_acc"], f"vc_macro_recall_{tag}": w["vc_macro_recall"],
                  f"n_cross_{tag}": w["n_crossing_notes"]})
    N = len(pred)
    r["density_imbalance"] = abs(int((pred == 0).sum()) - int((pred == 1).sum())) / N
    r["gt_density_imbalance"] = abs(int((g == 0).sum()) - int((g == 1).sum())) / N
    both = coll = 0
    for c in ex["clusters"]:
        if len(set(g[c])) == 2:
            both += 1
            coll += int(len(set(pred[c])) == 1)
    r["collapse_rate"] = coll / both if both else np.nan
    return r


# ===================================================================== B1
B1_GRID = [dict(w_reg=a, w_ergo=b, w_cross=c, w_empty=d, beam=e)
           for a, b, c, d, e in product([0.0, 0.5, 1.0, 2.0], [0.0, 0.3, 1.0], [0.0, 1.0, 3.0], [0.0, 0.5, 1.5], [1, 8])]


def _ergo_cost(S, prev, fam):
    if not S:
        return 0.0
    e = core.phi_ergo(tuple(S), tuple(prev), fam)
    return 10.0 if not np.isfinite(e) else -e


def b1_decode(ex, w_reg, w_ergo, w_cross, w_empty, beam, w_jump=1.0, ema=0.3):
    n, fam = ex["notes"], ex["family"]
    pitch = n.pitch.values
    # beam item: (cost, labels(list of arrays), reg[2], last[2])
    B = [(0.0, [], [None, None], [(), ()])]
    for c in ex["clusters"]:
        ps = pitch[c]
        new = []
        for cost, labs, reg, last in B:
            for lab in labelings(len(c)):
                S = [tuple(ps[lab == v]) for v in (0, 1)]
                add = 0.0
                for v in (0, 1):
                    if not S[v]:
                        add += w_empty
                        continue
                    mu = float(np.mean(S[v]))
                    if last[v]:
                        add += w_jump * abs(mu - np.mean(last[v])) / 12
                    if reg[v] is not None:
                        add += w_reg * abs(mu - reg[v]) / 12
                    add += w_ergo * _ergo_cost(S[v], last[v], fam)
                if S[0] and S[1] and reg[0] is not None and reg[1] is not None:
                    lo, hi = (0, 1) if reg[0] <= reg[1] else (1, 0)
                    add += w_cross * float(max(S[lo]) > min(S[hi]))
                nreg = [reg[v] if not S[v] else (np.mean(S[v]) if reg[v] is None else (1 - ema) * reg[v] + ema * np.mean(S[v]))
                        for v in (0, 1)]
                nlast = [S[v] if S[v] else last[v] for v in (0, 1)]
                new.append((cost + add, labs + [lab], nreg, nlast))
        new.sort(key=lambda x: x[0])
        B = new[:beam]
    pred = np.zeros(len(n), int)
    for c, lab in zip(ex["clusters"], B[0][1]):
        pred[c] = lab
    return pred


# ===================================================================== B2
B2_GRID = [dict(w_ac=a, w_p=b, stick=c, resid=d)
           for a, b, c, d in product([0.0, 0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.0, 0.5, 2.0], [True, False])]


def pitch_residualise(F, pitch):
    deg = int(min(3, len(np.unique(pitch)) - 1))
    if deg < 1:
        return F - F.mean(0)
    pc = (pitch - pitch.mean()) / (pitch.std() + 1e-9)
    V = np.vander(pc, deg + 1)
    beta, *_ = np.linalg.lstsq(V, F, rcond=None)
    return F - V @ beta


def b2_decode(ex, w_ac, w_p, stick, resid, n_iter=3, beam=4, seed=0):
    from sklearn.cluster import SpectralClustering
    from sklearn.decomposition import PCA
    n = ex["notes"]
    pitch = n.pitch.values.astype(float)
    on = n.onset.values
    A = pitch_residualise(ex["fz"], pitch) if resid else ex["fz"]
    A = PCA(n_components=6, random_state=seed).fit_transform(A)
    A = (A - A.mean(0)) / (A.std(0) + 1e-9)
    X = np.c_[w_ac * A, w_p * (pitch - pitch.mean()) / 6.0]
    D2 = ((X[:, None] - X[None]) ** 2).sum(-1) / X.shape[1]
    W = np.exp(-D2) * (np.abs(on[:, None] - on[None]) <= 2.0)
    np.fill_diagonal(W, 1.0)
    lab = SpectralClustering(2, affinity="precomputed", random_state=seed, assign_labels="discretize").fit_predict(W + 1e-6)
    for _ in range(n_iter):
        if len(set(lab)) < 2:
            break
        mu = [X[lab == v].mean(0) for v in (0, 1)]
        sd = [X[lab == v].std(0) + 0.3 for v in (0, 1)]
        ll = np.stack([(-0.5 * ((X - mu[v]) / sd[v]) ** 2 - np.log(sd[v])).sum(1) for v in (0, 1)], 1)
        Bm = [(0.0, [], [None, None])]
        for c in ex["clusters"]:
            new = []
            for cost, labs, last in Bm:
                for l in labelings(len(c)):
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
    return lab


# ===================================================================== B3
def b3_inputs(ex):
    n = ex["notes"]
    p = n.pitch.values.astype(float)
    on = n.onset.values
    csize = np.zeros(len(n))
    rank = np.zeros(len(n))
    for c in ex["clusters"]:
        csize[c] = len(c)
        rank[c] = np.argsort(np.argsort(p[c])) / max(1, len(c) - 1)
    dprev = np.r_[0, np.diff(on)]
    sym = np.c_[(p - p.mean()) / 12, (p - 60) / 24, np.log(n.duration.values + 1e-3), np.log(dprev + 1e-3),
                csize / 4, rank]
    cid = np.zeros(len(n), int)
    for k, c in enumerate(ex["clusters"]):
        cid[c] = k
    return dict(pitch=p.astype(int), sym=sym.astype(np.float32), ac=ex["fz"].astype(np.float32),
                onset=on.astype(np.float32), cid=cid, y=n.stream.values.astype(np.float32))


def make_b3(d=128, n_layer=3, n_head=4, n_sym=6, n_ac=31, dropout=0.1):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            self.l1, self.l2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.same_bias = nn.Parameter(torch.zeros(n_head))
            self.decay = nn.Parameter(torch.linspace(0.1, 2.0, n_head))
            self.drop = nn.Dropout(dropout)

        def forward(self, x, same, dt, mask):
            B, T, _ = x.shape
            q, k, v = self.qkv(self.l1(x)).view(B, T, 3, n_head, d // n_head).permute(2, 0, 3, 1, 4)
            bias = same[:, None] * self.same_bias[None, :, None, None] - dt[:, None] * F.softplus(self.decay)[None, :, None, None]
            bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, T, d)))
            return x + self.drop(self.ff(self.l2(x)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.pe = nn.Embedding(128, d)
            self.sym = nn.Linear(n_sym, d)
            self.ac = nn.Sequential(nn.Dropout(0.2), nn.Linear(n_ac, d))
            self.layers = nn.ModuleList([Layer() for _ in range(n_layer)])
            self.ln = nn.LayerNorm(d)
            self.node = nn.Linear(d, 1)
            self.pair = nn.Linear(d, 32)

        def forward(self, pitch, sym, ac, onset, cid, mask):
            x = self.pe(pitch) + self.sym(sym) + self.ac(ac)
            same = (cid[:, :, None] == cid[:, None, :]).float()
            dt = (onset[:, :, None] - onset[:, None, :]).abs()
            for l in self.layers:
                x = l(x, same, dt, mask)
            x = self.ln(x)
            u = self.pair(x)
            return self.node(x).squeeze(-1), torch.einsum("bik,bjk->bij", u, u) / np.sqrt(32)

    return Net()


def b3_loss(node, pair, y, onset, mask, w_pair=0.5, w_cont=0.1):
    """Permutation-invariant BCE (per excerpt, min over label swap) + pairwise same-stream
    BCE within 2 s (permutation-invariant by construction) + consistency regulariser."""
    import torch
    import torch.nn.functional as F
    m = mask.float()
    b1 = (F.binary_cross_entropy_with_logits(node, y, reduction="none") * m).sum(1) / m.sum(1)
    b2 = (F.binary_cross_entropy_with_logits(node, 1 - y, reduction="none") * m).sum(1) / m.sum(1)
    pit = torch.minimum(b1, b2).mean()
    same = (y[:, :, None] == y[:, None, :]).float()
    pm = (m[:, :, None] * m[:, None, :]) * ((onset[:, :, None] - onset[:, None, :]).abs() <= 2.0).float()
    pl = (F.binary_cross_entropy_with_logits(pair, same, reduction="none") * pm).sum() / pm.sum()
    p = torch.sigmoid(node)
    cons = p[:, :, None] * p[:, None, :] + (1 - p[:, :, None]) * (1 - p[:, None, :])
    cl = (F.binary_cross_entropy(cons.clamp(1e-4, 1 - 1e-4), torch.sigmoid(pair).detach(), reduction="none") * pm).sum() / pm.sum()
    return pit + w_pair * pl + w_cont * cl


def b3_collate(items, device, augment=False, rng=None):
    import torch
    T = max(len(it["pitch"]) for it in items)
    B = len(items)
    out = {k: np.zeros((B, T), np.int64) for k in ("pitch", "cid")}
    out.update(sym=np.zeros((B, T, 6), np.float32), ac=np.zeros((B, T, 31), np.float32),
               onset=np.zeros((B, T), np.float32), y=np.zeros((B, T), np.float32), mask=np.zeros((B, T), bool))
    for b, it in enumerate(items):
        n = len(it["pitch"])
        sh = int(rng.integers(-3, 4)) if augment else 0
        out["pitch"][b, :n] = np.clip(it["pitch"] + sh, 0, 127)
        s = it["sym"].copy()
        s[:, 1] += sh / 24
        out["sym"][b, :n] = s
        out["ac"][b, :n] = it["ac"]
        out["onset"][b, :n] = it["onset"]
        out["cid"][b, :n] = it["cid"]
        out["cid"][b, n:] = -1 - np.arange(T - n)
        out["y"][b, :n] = it["y"]
        out["mask"][b, :n] = True
    return {k: torch.from_numpy(v).to(device) for k, v in out.items()}


def b3_train(train_items, device, epochs=150, batch=8, lr=1e-3, seed=0):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_b3().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(train_items))
        for i in range(0, len(order), batch):
            b = b3_collate([train_items[j] for j in order[i:i + batch]], device, augment=True, rng=rng)
            node, pair = net(b["pitch"], b["sym"], b["ac"], b["onset"], b["cid"], b["mask"])
            loss = b3_loss(node, pair, b["y"], b["onset"], b["mask"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
    return net


def b3_predict(nets, item, device):
    """Average node probabilities over nets after aligning each net's labelling to the
    first (permutation-invariant ensembling); threshold 0.5."""
    import torch
    b = b3_collate([item], device)
    probs = []
    with torch.no_grad():
        for net in nets:
            net.eval()
            node, _ = net(b["pitch"], b["sym"], b["ac"], b["onset"], b["cid"], b["mask"])
            p = torch.sigmoid(node)[0].cpu().numpy()
            if probs and np.mean(np.abs(p - probs[0])) > np.mean(np.abs((1 - p) - probs[0])):
                p = 1 - p
            probs.append(p)
    return (np.mean(probs, 0) > 0.5).astype(int)


# ===================================================================== B4
def b4_decode(ex, prior, ergo_prune=True):
    """Greedy naive LM decoding: argmax_S log P(S|H0) + log P(C\\S|H1)."""
    from generative_prior import step_tokens
    n, fam = ex["notes"], ex["family"]
    pitch, dur, on = n.pitch.values, n.duration.values, n.onset.values
    hist = [[], []]
    last_t, last_set = None, [(), ()]
    pred = np.zeros(len(n), int)
    for c in ex["clusters"]:
        t = float(on[c[0]])
        notes = [(int(pitch[i]), float(dur[i])) for i in c]
        labs = labelings(len(c))
        subsets = sorted({tuple(sorted(notes[i] for i in np.where(l == 1)[0])) for l in labs} |
                         {tuple(sorted(notes[i] for i in np.where(l == 0)[0])) for l in labs})
        sidx = {s: k for k, s in enumerate(subsets)}
        ctx = [prior.context(fam, hist[v], last_t, t) for v in (0, 1)]
        lp = prior.logp_contexts(ctx, subsets)
        best, best_s = None, -np.inf
        for l in labs:
            S = [tuple(sorted(notes[i] for i in np.where(l == v)[0])) for v in (0, 1)]
            s = lp[0, sidx[S[0]]] + lp[1, sidx[S[1]]]
            if ergo_prune:
                e = [core.phi_ergo(tuple(p for p, _ in S[v]), last_set[v], fam) for v in (0, 1)]
                if not all(np.isfinite(e)):
                    s -= 1e6
            if s > best_s:
                best_s, best = s, l
        pred[c] = best
        for v in (0, 1):
            S = [notes[i] for i in np.where(best == v)[0]]
            hist[v] += step_tokens(4.0 if last_t is None else t - last_t, S)
            if S:
                last_set[v] = tuple(sorted(p for p, _ in S))
        last_t = t
    return pred
