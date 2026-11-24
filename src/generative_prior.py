"""Neural polyphonic symbolic-music prior P_theta(S_t | H_t) (Step 3).

A causal Transformer LM pretrained on single guitar / keyboard / bass tracks of the Lakh
clean_midi corpus, EXCLUDING every benchmark piece group (dev and confirmation; title-
normalised) and every track whose content signature matches a benchmark source track.

Tokenisation of one stream on a merged onset grid (target track + one companion track,
30 ms clustering), so empty steps occur exactly as in duet decoding:
    [BOS] [FAM_x] { TS_k  (PITCH_p DUR_d)*  END }*
TS_k  : log-binned time since the previous grid step (32 bins, 20 ms .. 4 s)
PITCH : 128 pitches, ascending within a step;  DUR: 16 log bins (30 ms .. 4 s)
log P(S_t | H_t) = sum of log-probs of the continuation (PITCH DUR)* END after the
context [BOS FAM history TS_t]. TS tokens are context only (excluded from the loss).

CLI:
  python workflow/generative_prior.py build-corpus --workers 16
  python workflow/generative_prior.py train --steps 5000 --device cuda:0
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SESSION = Path(__file__).resolve().parents[1]
CKPT = SESSION / "checkpoints"
sys.path.insert(0, str(Path(__file__).resolve().parent))

PAD, BOS, END = 0, 1, 2
FAM_TOK = {"guitar": 3, "keyboard": 4, "bass": 5}
TS0, N_TS = 6, 32
P0 = TS0 + N_TS            # 38
D0 = P0 + 128              # 166
N_D = 16
VOCAB = D0 + N_D           # 182
TS_EDGES = np.logspace(np.log10(0.02), np.log10(4.0), N_TS - 1)
D_EDGES = np.logspace(np.log10(0.03), np.log10(4.0), N_D - 1)
CTX = 512
MAX_HIST = 440             # history tokens kept (truncated at a step boundary)
CLUSTER_TOL = 0.03
FAMILY_PROGRAMS = {"keyboard": range(0, 8), "guitar": range(24, 32), "bass": range(32, 40)}
SEED = 42


def log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def ts_token(dt_s: float) -> int:
    return TS0 + int(np.searchsorted(TS_EDGES, max(dt_s, 0.0)))


def dur_token(d: float) -> int:
    return D0 + int(np.searchsorted(D_EDGES, max(d, 0.0)))


def step_tokens(dt_s: float, notes) -> list[int]:
    """notes: iterable of (pitch, duration_s)."""
    out = [ts_token(dt_s)]
    for p, d in sorted(notes):
        out += [P0 + int(p), dur_token(d)]
    return out + [END]


def cont_tokens(notes) -> list[int]:
    out = []
    for p, d in sorted(notes):
        out += [P0 + int(p), dur_token(d)]
    return out + [END]


def grid_times(onsets, tol=CLUSTER_TOL):
    o = np.sort(np.asarray(onsets, float))
    g = []
    for t in o:
        if not g or t - g[-1] > tol:
            g.append(t)
    return np.array(g)


def track_tokens(fam, notes, companion_onsets):
    """notes: (n,3) [onset, duration, pitch] of the target track."""
    g = grid_times(np.concatenate([notes[:, 0], companion_onsets]))
    idx = np.searchsorted(g, notes[:, 0] + 1e-9, side="right") - 1
    toks = [BOS, FAM_TOK[fam]]
    prev = None
    for k, t in enumerate(g):
        sel = notes[idx == k]
        toks += step_tokens(4.0 if prev is None else t - prev, [(int(p), float(d)) for _, d, p in sel])
        prev = t
    return toks


def signature(notes):
    """Content signature of a track: first 48 (10 ms-quantised relative onset, pitch)."""
    a = notes[np.lexsort((notes[:, 2], notes[:, 0]))][:48]
    if not len(a):
        return ""
    rel = np.round((a[:, 0] - a[0, 0]) * 100).astype(int)
    return hashlib.sha1(np.c_[rel, a[:, 2].astype(int)].tobytes()).hexdigest()


def _norm_title(t):
    t = re.sub(r"\.\d+$", "", t)
    t = re.sub(r"\(.*?\)|\[.*?\]", "", t)
    return re.sub(r"[^a-z0-9]", "", t.lower())


# ===================================================================== corpus
def _benchmark_exclusions():
    """Piece groups + track signatures of ALL benchmark source tracks (both splits).
    Uses only manifest metadata and the raw source MIDI files (never workflow/data/confirmation)."""
    import pretty_midi
    man = json.loads((SESSION / "results" / "dataset_manifest.json").read_text())
    groups, sigs = set(), set()
    for e in man["excerpts"]:
        groups.add(e["piece_group"])
        if e["source"] == "lakh_clean_midi":
            pm = pretty_midi.PrettyMIDI(str(SESSION / e["source_files"][0]))
            for sid in e["stem_ids"]:
                t = int(sid.split("#track")[1])
                ins = pm.instruments[t]
                sigs.add(signature(np.array([(n.start, n.end - n.start, n.pitch) for n in ins.notes], float)))
    return groups, sigs


def _corpus_file(args):
    import pretty_midi
    path, seed = args
    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception:  # noqa: BLE001
        return []
    if not np.isfinite(pm.get_end_time()) or pm.get_end_time() > 1200:
        return []
    rng = np.random.default_rng(seed)
    tracks = [(i, ins) for i, ins in enumerate(pm.instruments) if not ins.is_drum and len(ins.notes) >= 20]
    out = []
    for i, ins in tracks:
        fam = next((f for f, r in FAMILY_PROGRAMS.items() if ins.program in r), None)
        if fam is None:
            continue
        notes = np.array([(n.start, n.end - n.start, n.pitch) for n in ins.notes], float)
        notes = notes[np.lexsort((notes[:, 2], notes[:, 0]))]
        same = [j for j, o in tracks if j != i and o.program in FAMILY_PROGRAMS[fam]]
        other = [j for j, _ in tracks if j != i]
        comp = int(rng.choice(same)) if same else (int(rng.choice(other)) if other else None)
        comp_on = np.array([n.start for n in pm.instruments[comp].notes]) if comp is not None else np.zeros(0)
        toks = track_tokens(fam, notes, comp_on)
        out.append(dict(fam=fam, toks=np.array(toks[:6000], np.int16), sig=signature(notes)))
    return out


def build_corpus(workers):
    CKPT.mkdir(parents=True, exist_ok=True)
    groups, sigs = _benchmark_exclusions()
    log(f"Excluding {len(groups)} benchmark piece groups and {len(sigs)} benchmark track signatures")
    root = SESSION / "data" / "raw" / "lmd" / "x" / "clean_midi"
    files = sorted(root.rglob("*.mid"))
    keep, n_excl_group = [], 0
    for f in files:
        if "lakh:" + _norm_title(f.stem) in groups:
            n_excl_group += 1
            continue
        keep.append(f)
    log(f"{len(files)} files; excluded by piece group: {n_excl_group}; parsing {len(keep)}")
    jobs = [(str(f), SEED + i) for i, f in enumerate(keep)]
    rows, t0 = [], time.time()
    with Pool(workers) as pool:
        for i, (f, r) in enumerate(zip(keep, pool.imap(_corpus_file, jobs, chunksize=32))):
            g = _norm_title(f.stem)
            for x in r:
                x["group"] = g
                x["file"] = str(f.relative_to(SESSION))
            rows.extend(r)
            if (i + 1) % 3000 == 0:
                log(f"   {i + 1}/{len(keep)} files, {len(rows)} tracks ({time.time() - t0:.0f}s)")
    n0 = len(rows)
    rows = [r for r in rows if r["sig"] not in sigs]
    n_sig = n0 - len(rows)
    seen, dedup = set(), []
    for r in rows:
        if r["sig"] in seen:
            continue
        seen.add(r["sig"])
        dedup.append(r)
    log(f"tracks: {n0}; removed by benchmark signature: {n_sig}; after de-dup: {len(dedup)}")
    # validation split by group hash (2%)
    val = np.array([int(hashlib.md5(r["group"].encode()).hexdigest(), 16) % 50 == 0 for r in dedup])
    toks = np.concatenate([r["toks"] for r in dedup])
    lens = np.array([len(r["toks"]) for r in dedup])
    offs = np.r_[0, np.cumsum(lens)[:-1]]
    fams = np.array([r["fam"] for r in dedup])
    np.savez_compressed(CKPT / "prior_corpus.npz", toks=toks, offs=offs, lens=lens, fams=fams, val=val,
                        groups=np.array([r["group"] for r in dedup]))
    meta = dict(n_files_total=len(files), n_files_excluded_piece_group=n_excl_group, n_tracks_parsed=n0,
                n_tracks_removed_benchmark_signature=n_sig, n_tracks=len(dedup), n_tokens=int(lens.sum()),
                n_val_tracks=int(val.sum()), family_counts={f: int((fams == f).sum()) for f in FAM_TOK},
                excluded_piece_groups=len(groups), excluded_signatures=len(sigs),
                benchmark_groups_in_corpus=int(sum(("lakh:" + r["group"]) in groups for r in dedup)),
                benchmark_signatures_in_corpus=int(sum(r["sig"] in sigs for r in dedup)))
    (CKPT / "prior_corpus_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"corpus: {meta}")


# ===================================================================== model
def _torch():
    import torch
    return torch


def make_model(d=256, n_layer=6, n_head=8, dropout=0.1):
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            B, T, _ = x.shape
            q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, n_head, d // n_head).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=dropout if self.training else 0.0)
            x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, d)))
            return x + self.drop(self.mlp(self.ln2(x)))

    class CausalTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(CTX, d)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.ln = nn.LayerNorm(d)
            self.head = nn.Linear(d, VOCAB, bias=False)
            self.head.weight = self.tok.weight

        def forward(self, idx):
            x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
            for b in self.blocks:
                x = b(x)
            return self.head(self.ln(x))

    return CausalTransformer()


def _batch(toks, offs, lens, sel, rng, B, step_pos_cache):
    """Random crops of CTX tokens: [BOS FAM] + tokens starting at a TS token."""
    out = np.zeros((B, CTX), np.int64)
    w = lens[sel].astype(float)
    pick = rng.choice(sel, size=B, p=w / w.sum())
    for b, i in enumerate(pick):
        seq = toks[offs[i]: offs[i] + lens[i]]
        if i not in step_pos_cache:
            step_pos_cache[i] = np.where((seq >= TS0) & (seq < P0))[0]
        sp = step_pos_cache[i]
        cands = sp[sp <= max(sp[0], len(seq) - 64)]
        s = int(rng.choice(cands))
        chunk = np.r_[seq[:2], seq[s: s + CTX - 2]]
        out[b, :len(chunk)] = chunk
    return out


def train(steps, device, batch=64, lr=6e-4):
    torch = _torch()
    import torch.nn.functional as F
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    z = np.load(CKPT / "prior_corpus.npz")
    toks, offs, lens, val = z["toks"].astype(np.int64), z["offs"], z["lens"], z["val"]
    tr_idx, va_idx = np.where(~val)[0], np.where(val)[0]
    model = make_model().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"model params: {n_par / 1e6:.2f}M; train tracks {len(tr_idx)}, val tracks {len(va_idx)}; device {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")
    cache = {}
    va_rng = np.random.default_rng(SEED + 1)
    va_batches = [_batch(toks, offs, lens, va_idx, va_rng, batch, {}) for _ in range(20)]

    def lossf(x):
        logits = model(x[:, :-1])
        tgt = x[:, 1:].clone()
        tgt[(tgt >= TS0) & (tgt < P0)] = PAD          # TS tokens are context only
        return F.cross_entropy(logits.float().reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=PAD)

    hist, t0 = [], time.time()
    for it in range(1, steps + 1):
        cur_lr = lr * min(1.0, it / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, it / steps)))
        for g in opt.param_groups:
            g["lr"] = max(cur_lr, lr * 0.02)
        model.train()
        x = torch.from_numpy(_batch(toks, offs, lens, tr_idx, rng, batch, cache)).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(x)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if it % 250 == 0 or it == steps:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                vl = float(np.mean([lossf(torch.from_numpy(v).to(device)).item() for v in va_batches]))
            hist.append(dict(step=it, train_loss=float(loss.item()), val_loss=vl, elapsed_s=time.time() - t0))
            log(f"step {it}/{steps} train {loss.item():.3f} val {vl:.3f} ({time.time() - t0:.0f}s)")
    torch.save(model.state_dict(), CKPT / "prior_lm.pt")
    meta = dict(arch=dict(d=256, n_layer=6, n_head=8, ctx=CTX, vocab=VOCAB), n_params=n_par, steps=steps, batch=batch,
                lr=lr, seed=SEED, history=hist, corpus=json.loads((CKPT / "prior_corpus_meta.json").read_text()),
                written_at=dt.datetime.now(dt.timezone.utc).isoformat())
    (CKPT / "prior_lm_meta.json").write_text(json.dumps(meta, indent=2))


# ===================================================================== inference API
from decgd import GenerativePriorModel  # noqa: E402


class NeuralPolyphonicPrior(GenerativePriorModel):
    """Batched log P(S | H) from the pretrained LM (implements decgd.GenerativePriorModel)."""

    def __init__(self, device=None, ckpt=CKPT / "prior_lm.pt"):
        torch = _torch()
        self.torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = make_model(dropout=0.0).to(self.device)
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()
        self.n_calls = 0

    @staticmethod
    def history_tokens(family, history):
        """history: list of (onset_s, notes) on the merged grid -> token list."""
        toks, prev = [], None
        for t, notes in history:
            toks += step_tokens(4.0 if prev is None else t - prev, notes)
            prev = t
        return toks

    def context(self, family, hist_toks, last_t, onset_s):
        ts = ts_token(4.0 if last_t is None else onset_s - last_t)
        h = hist_toks
        if len(h) > MAX_HIST:                      # truncate at a step boundary
            cut = len(h) - MAX_HIST
            while cut < len(h) and not (TS0 <= h[cut] < P0):
                cut += 1
            h = h[cut:]
        return [BOS, FAM_TOK[family]] + list(h) + [ts]

    def logp_contexts(self, contexts, candidates):
        """log P(cand | ctx) for every (ctx, cand) pair -> array (len(contexts), len(candidates))."""
        pairs = [(c, S) for c in contexts for S in candidates]
        return self.logp_pairs(pairs).reshape(len(contexts), len(candidates))

    def logp_pairs(self, pairs):
        """Batched log P(S | ctx) for an explicit list of (context_tokens, candidate_notes) pairs."""
        torch = self.torch
        seqs, meta = [], []
        for c, S in pairs:
            s = cont_tokens(S)
            seqs.append(list(c) + s)
            meta.append((len(c), len(s)))
        out = np.zeros(len(seqs))
        order = np.argsort([len(s) for s in seqs])          # length-bucketed batches
        with torch.no_grad():
            for b0 in range(0, len(seqs), 256):
                idx = order[b0:b0 + 256]
                L = max(len(seqs[i]) for i in idx)
                x = np.zeros((len(idx), L), np.int64)
                for j, i in enumerate(idx):
                    x[j, :len(seqs[i])] = seqs[i]
                xb = torch.from_numpy(x).to(self.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                    lp = torch.log_softmax(self.model(xb).float(), -1)
                tgt = torch.from_numpy(np.roll(x, -1, 1)).to(self.device)
                g = lp.gather(-1, tgt[..., None])[..., 0].cpu().numpy()
                for j, i in enumerate(idx):
                    lc, ls = meta[i]
                    out[i] = g[j, lc - 1: lc - 1 + ls].sum()
        self.n_calls += 1
        return out

    def logp_candidates(self, family, history, onset_s, candidates):
        last_t = history[-1][0] if history else None
        ctx = self.context(family, self.history_tokens(family, history), last_t, onset_s)
        return self.logp_contexts([ctx], candidates)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-corpus", "train"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.cmd == "build-corpus":
        build_corpus(a.workers)
    else:
        train(a.steps, a.device)
