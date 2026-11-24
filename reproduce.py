"""One-click reproduction of the held-out same-timbre guitar-duet evaluation of DECGD (dual-stream equilibrium contrastive
generative decoding): two-stream note separation of rhythm / lead guitar duets played with one guitar timbre.

Steps
  1. Integrity: SHA-256 of every bundled source / data / checkpoint / config file against integrity_lock.json.
  2. Metrics: per-excerpt permutation-invariant Macro-F1 (PI-Macro-F1), permutation-invariant note accuracy (PI-Acc),
     voice swap error rate (VSER), accuracy on notes in dense (>= 3-note) onset clusters, and stream imbalance error
     |Imb_pred - Imb_gt|, recomputed from the cached predictions and the ground-truth notes; checked against the bundled
     reference per-excerpt metrics and the expected values of the three subsets.
  3. Statistics: two-sided paired Wilcoxon signed-rank tests at excerpt and piece-group level, Holm step-down correction
     within each comparison family, matched-pairs rank-biserial r, and 10,000-resample piece-group block bootstrap 95% CIs;
     for all guitar duets (N = 69, G = 48), real GuitarSet duets (N = 14, G = 7) and rendered Lakh guitar duets
     (N = 55, G = 41).
  4. Live decoding check (skip with --skip-decode-check): B3 single model, 6-model B3 ensemble, B2 soft posterior, a
     prior-free DECGD configuration and the full DECGD decoder are run on two rendered and one real guitar excerpt and
     compared with the cached predictions (exact label agreement up to a global relabelling).
Outputs: outputs/main_results.{csv,json}, outputs/ablation_results.{csv,json}, outputs/per_excerpt_metrics.csv,
         outputs/verification_report.json
Exit code 0 only if every check passes.
Usage: python3 reproduce.py [--skip-decode-check] [--device cpu|cuda:0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
SEED = 42
STRATA = ["GuitarSet guitar", "Lakh guitar"]
SUBSETS = ["guitar_all"] + STRATA
N_EXPECTED = {"guitar_all": (69, 48), "GuitarSet guitar": (14, 7), "Lakh guitar": (55, 41)}
FULL, ENS, SINGLE = "DECGD_full", "B3_ensemble_only", "B3_single_model"
BASELINES = ["B1_viterbi_dp", "B2_acoustic_symbolic_hmm", SINGLE, "B4_naive_lm_prior", "decgd_classic", "decgd_hybrid_full",
             "decgd_hybrid_all"]
CONSTRUCTIVE = ["ens_decoder_only", "ens_plus_capdebias", "ens_plus_b2", "ens_plus_genprior", "ens_plus_capdebias_b2",
                "ens_plus_capdebias_genprior"]
MULTI = ["DECGD_wo_capacity", "DECGD_wo_prior_ergo"]
LOO = ["DECGD_wo_b3_debias", "DECGD_wo_b2", "DECGD_wo_b3", "DECGD_wo_genprior", "DECGD_wo_capacity_eq", "DECGD_wo_ergo"]
ABLATION_FAMILY = [ENS] + CONSTRUCTIVE + MULTI
METHODS = BASELINES + [ENS] + CONSTRUCTIVE + MULTI + LOO + [FULL]
METRICS = ["pi_macro_f1", "pi_acc", "swap_rate", "imb_err", "acc_c3p"]
TESTED = ["pi_macro_f1", "pi_acc", "imb_err", "swap_rate", "acc_c3p"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ 1. integrity
def check_integrity():
    lock = json.loads((ROOT / "integrity_lock.json").read_text())
    bad = [p for p, h in lock.items() if not (ROOT / p).exists() or sha(ROOT / p) != h]
    return dict(n_files=len(lock), n_mismatch=len(bad), mismatches=bad, passed=not bad)


# ------------------------------------------------------------------ 2. metrics
def load_benchmark():
    import baselines as BL
    idx = pd.read_csv(ROOT / "data" / "heldout_index.csv")
    notes = pd.read_csv(ROOT / "data" / "heldout_notes.csv")
    EX = {}
    for r in idx.itertuples():
        n = notes[notes.excerpt_id == r.excerpt_id].drop(columns="excerpt_id").reset_index(drop=True)
        EX[r.excerpt_id] = dict(id=r.excerpt_id, group=r.piece_group, stratum=r.stratum, source=r.source, family=r.family,
                                notes=n, clusters=BL.cluster_index(n.onset.values), split="heldout")
    P = pd.read_csv(ROOT / "data" / "heldout_predictions.csv")
    return idx, EX, {(r.excerpt_id, r.method): np.array(json.loads(r.pred), int) for r in P.itertuples()}


def size_acc(ex, pred):
    g = ex["notes"].stream.values
    if np.mean(pred == g) < np.mean((1 - pred) == g):
        pred = 1 - pred
    size = np.zeros(len(g), int)
    for c in ex["clusters"]:
        size[c] = len(c)
    m = size >= 3
    return dict(n_c3p=int(m.sum()), correct_c3p=int((pred[m] == g[m]).sum()), acc_c3p=float(np.mean(pred[m] == g[m])) if m.any() else np.nan)


def metric_table(EX, P):
    import baselines as BL
    rows = []
    for eid, ex in EX.items():
        for m in METHODS:
            pr = P[(eid, m)]
            r = BL.evaluate(ex, pr)
            rows.append(dict(excerpt_id=eid, method=m, piece_group=ex["group"], stratum=ex["stratum"], source=ex["source"],
                             pi_macro_f1=r["pi_macro_f1"], pi_acc=r["pi_acc"], swap_rate=r["swap_rate"],
                             imb_err=abs(r["density_imbalance"] - r["gt_density_imbalance"]), **size_acc(ex, pr)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. statistics
def summarise(D, m):
    import paired_stats as C
    g = D[D.method == m]
    s = {k: C.describe(g[k].values.astype(float), g.piece_group.values) for k in METRICS}
    for k in METRICS:
        s[k]["sd"] = float(np.nanstd(g[k].values.astype(float), ddof=1)) if g[k].notna().sum() > 1 else float("nan")
    s["pooled_acc_c3p"] = float(g.correct_c3p.sum() / g.n_c3p.sum()) if g.n_c3p.sum() else float("nan")
    return s


def family(D, a, others, k):
    import paired_stats as C
    d = D[D.n_c3p > 0] if k == "acc_c3p" else D
    return C.comparison_family(d, a, [o for o in others if o in set(d.method)], k)


def config_vs_ensemble(D, k):
    """Each constructive configuration minus the ensemble (Holm over 6); computed as ensemble - config, then sign-flipped."""
    rows = family(D, ENS, CONSTRUCTIVE, k)
    for r in rows:
        r.update(a=r["b"], b=ENS, mean_diff=-r["mean_diff"], median_diff=-r["median_diff"],
                 ci95_grouped_bootstrap=[-r["ci95_grouped_bootstrap"][1], -r["ci95_grouped_bootstrap"][0]],
                 ci95_excerpt_bootstrap=[-r["ci95_excerpt_bootstrap"][1], -r["ci95_excerpt_bootstrap"][0]],
                 rank_biserial=-r["rank_biserial"], rank_biserial_group=-r["rank_biserial_group"], cohens_dz=-r["cohens_dz"],
                 mean_diff_group=-r["mean_diff_group"])
    return rows


def analyse_subset(D):
    out = dict(n_excerpts=int(D.excerpt_id.nunique()), n_groups=int(D.piece_group.nunique()),
               summary={m: summarise(D, m) for m in METHODS}, tests={})
    for k in TESTED:
        out["tests"][k] = dict(full_vs_baselines=family(D, FULL, BASELINES, k),
                               full_vs_baselines_and_ensemble=family(D, FULL, BASELINES + [ENS], k),
                               full_vs_ensemble_and_ablations=family(D, FULL, ABLATION_FAMILY, k),
                               full_vs_single_module_ablations=family(D, FULL, LOO, k),
                               ensemble_vs_single=family(D, ENS, [SINGLE], k),
                               constructive_vs_ensemble=config_vs_ensemble(D, k))
    dec = {}
    for k, better in (("pi_macro_f1", 1), ("pi_acc", 1), ("imb_err", -1)):
        a, b, c = (out["summary"][m][k]["mean"] for m in (SINGLE, ENS, FULL))
        tot = c - a
        ok = better * tot > 1e-9
        dec[k] = dict(single=a, ensemble=b, full=c, total_change=tot, change_from_ensembling=b - a, change_from_decoding=c - b,
                      share_ensembling=(b - a) / tot if ok else None, share_decoding=(c - b) / tot if ok else None)
    out["gain_decomposition"] = dec
    return out


def rows_for(name, A, methods, fam_key, holm_tag):
    rows = []
    T = {k: {r["b"]: r for r in A["tests"][k][fam_key]} for k in ("pi_macro_f1", "pi_acc", "imb_err") if k in A["tests"]}
    for m in methods:
        s = A["summary"][m]
        r = dict(subset=name, method=m, n=A["n_excerpts"], n_groups=A["n_groups"], holm_family=holm_tag if m != FULL else "")
        for k in ("pi_macro_f1", "pi_acc"):
            r.update({k: s[k]["mean"], f"{k}_sd": s[k]["sd"], f"{k}_median": s[k]["median"], f"{k}_iqr_lo": s[k]["iqr"][0],
                      f"{k}_iqr_hi": s[k]["iqr"][1], f"{k}_ci_lo": s[k]["ci95_grouped_bootstrap"][0],
                      f"{k}_ci_hi": s[k]["ci95_grouped_bootstrap"][1]})
        r.update(vser=s["swap_rate"]["mean"], imb_err=s["imb_err"]["mean"], imb_err_ci_lo=s["imb_err"]["ci95_grouped_bootstrap"][0],
                 imb_err_ci_hi=s["imb_err"]["ci95_grouped_bootstrap"][1], dense_acc_pooled=s["pooled_acc_c3p"],
                 dense_acc_excerpt_mean=s["acc_c3p"]["mean"])
        for k, tag in (("pi_macro_f1", "f1"), ("pi_acc", "acc"), ("imb_err", "imb")):
            x = T.get(k, {}).get(m)
            if x:
                r.update({f"d{tag}_full_minus_method": x["mean_diff"], f"d{tag}_ci_lo": x["ci95_grouped_bootstrap"][0],
                          f"d{tag}_ci_hi": x["ci95_grouped_bootstrap"][1], f"p_{tag}_excerpt_raw": x["p_excerpt"],
                          f"p_{tag}_group_raw": x["p_group"], f"p_{tag}_excerpt_holm": x["p_excerpt_holm"],
                          f"p_{tag}_group_holm": x["p_group_holm"], f"rrb_{tag}": x["rank_biserial"]})
        rows.append(r)
    return rows


# ------------------------------------------------------------------ 4. live decoding check
def decode_check(EX, P, device):
    import torch
    import baselines as BL
    import decgd_v3 as V3
    from generative_prior import NeuralPolyphonicPrior
    torch.manual_seed(SEED)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() and torch.cuda.mem_get_info(0)[0] > 2.5e9 else "cpu"
    cfg = json.loads((ROOT / "configs" / "decode_check_specs.json").read_text())
    nets = []
    for ck in [f"b3_fold{k}.pt" for k in range(5)] + ["b3_full_dev.pt"]:
        net = BL.make_b3().to(device)
        net.load_state_dict(torch.load(ROOT / "checkpoints" / ck, map_location=device))
        net.eval()
        nets.append(net)
    res = []
    for eid in cfg["excerpts"]:
        t0 = time.time()
        ex = dict(EX[eid], fz=np.load(ROOT / "data" / "decode_check" / f"{eid}_features_z.npy").astype(np.float64))
        b = BL.b3_collate([BL.b3_inputs(ex)], device)
        ps, zs = [], []
        with torch.no_grad():
            for net in nets:
                node, pair = net(b["pitch"], b["sym"], b["ac"], b["onset"], b["cid"], b["mask"])
                ps.append(torch.sigmoid(node)[0].cpu().numpy().astype(np.float64))
                zs.append(pair[0].cpu().numpy().astype(np.float64))
        ref = ps[-1]
        al = [1 - p if np.mean(np.abs(p - ref)) > np.mean(np.abs((1 - p) - ref)) else p for p in ps]
        ens = dict(p=np.mean(al, 0), pair=np.mean(zs, 0))
        p = np.clip(ens["p"], V3.PCLIP, 1 - V3.PCLIP)
        z, _ = V3.chunk_align(np.log(p) - np.log(1 - p), ens["pair"], ex["notes"].onset.values, chunk_s=8.0)
        live = {SINGLE: (ref > 0.5).astype(int), ENS: (z > 0).astype(int)}
        lab2, ll2 = V3.b2_soft(ex, **cfg["b2_soft"])
        live["B2_acoustic_symbolic_hmm"] = lab2
        for m, sp in cfg["specs"][eid].items():
            x = V3.attach_v3(ex, ens, dict(lab=lab2, ll=ll2), V3.V3Inputs(**sp["inputs"]))
            prior = NeuralPolyphonicPrior(device=device, ckpt=ROOT / "checkpoints" / f"{sp['prior']}.pt") \
                if (sp["decoder"].get("w_swap") or sp["decoder"].get("w_lm")) else None
            live[m] = np.asarray(V3.decode_many(prior, x, {m: sp["decoder"]})[m][0], int)
        for m, y in live.items():
            c = P[(eid, m)]
            res.append(dict(excerpt_id=eid, stratum=EX[eid]["stratum"], method=m, n_notes=int(len(y)),
                            identical=bool(np.array_equal(y, c) or np.array_equal(1 - y, c)),     # up to a global 0/1 relabelling
                            same_label_names=bool(np.array_equal(y, c)),
                            n_label_diff_up_to_relabel=int(min(np.sum(y != c), np.sum((1 - y) != c)))))
        log(f"   {eid} ({EX[eid]['stratum']}, {len(ex['notes'])} notes): "
            f"{sum(r['identical'] for r in res if r['excerpt_id'] == eid)}/{len(live)} identical ({time.time() - t0:.0f} s)")
    return dict(device=device, criterion="exact agreement up to a global relabelling of the two streams (all metrics are "
                "permutation invariant)", results=res, passed=all(r["identical"] for r in res))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-decode-check", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    rep = dict(seed=SEED, benchmark="held-out same-timbre guitar duets",
               environment=dict(python=platform.python_version(), platform=platform.platform()))
    for mod in ("numpy", "pandas", "scipy", "statsmodels", "sklearn", "torch"):
        try:
            rep["environment"][mod] = __import__(mod).__version__
        except ImportError:
            rep["environment"][mod] = None
    log("1/4 integrity check")
    rep["integrity"] = check_integrity()
    if not rep["integrity"]["passed"]:
        (OUT / "verification_report.json").write_text(json.dumps(rep, indent=1))
        sys.exit(f"integrity check failed: {rep['integrity']['mismatches'][:5]}")
    log("2/4 per-excerpt metrics")
    idx, EX, P = load_benchmark()
    assert set(idx.stratum) == set(STRATA) and (idx.family == "guitar").all()
    D = metric_table(EX, P)
    D.to_csv(OUT / "per_excerpt_metrics.csv", index=False)
    R = pd.read_csv(ROOT / "data" / "reference_per_excerpt_metrics.csv").merge(D, on=["excerpt_id", "method"], suffixes=("_ref", ""))
    diffs = {k: float(np.nanmax(np.abs(R[k] - R[f"{k}_ref"]))) for k in METRICS}
    rep["per_excerpt_recompute"] = dict(n_rows=int(len(R)), max_abs_diff=diffs,
                                        passed=bool(len(R) == len(idx) * len(METHODS) and max(diffs.values()) < 1e-12))
    log("3/4 statistics (10,000-resample bootstraps)")
    subsets = {"guitar_all": D, **{s: D[D.stratum == s] for s in STRATA}}
    A = {}
    for name, d in subsets.items():
        A[name] = analyse_subset(d)
        log(f"   {name}: N={A[name]['n_excerpts']} G={A[name]['n_groups']} F1 full {A[name]['summary'][FULL]['pi_macro_f1']['mean']:.3f} "
            f"ensemble {A[name]['summary'][ENS]['pi_macro_f1']['mean']:.3f} single {A[name]['summary'][SINGLE]['pi_macro_f1']['mean']:.3f}")
    exp = json.loads((ROOT / "data" / "expected_headline_values.json").read_text())
    hd = 0.0
    for name, e in exp.items():
        S = A[name]["summary"]
        for m, vals in e.items():
            for k, v in vals.items():
                got = S[m]["pooled_acc_c3p"] if k == "pooled_acc_c3p" else S[m][k]["mean"]
                hd = max(hd, abs(got - v)) if np.isfinite(v) else hd
    sizes_ok = all((A[n]["n_excerpts"], A[n]["n_groups"]) == N_EXPECTED[n] for n in SUBSETS)
    rep["headline_values"] = dict(
        {n: dict(full_pi_macro_f1=A[n]["summary"][FULL]["pi_macro_f1"]["mean"], full_pi_acc=A[n]["summary"][FULL]["pi_acc"]["mean"],
                 full_imb_err=A[n]["summary"][FULL]["imb_err"]["mean"], ensemble_pi_macro_f1=A[n]["summary"][ENS]["pi_macro_f1"]["mean"],
                 single_pi_macro_f1=A[n]["summary"][SINGLE]["pi_macro_f1"]["mean"]) for n in SUBSETS},
        subset_sizes={n: [A[n]["n_excerpts"], A[n]["n_groups"]] for n in SUBSETS}, subset_sizes_ok=sizes_ok,
        max_abs_diff_vs_expected=hd, passed=bool(hd < 1e-12 and sizes_ok))
    main_json = dict(benchmark="held-out same-timbre guitar duets",
                     subsets={n: dict(n_excerpts=A[n]["n_excerpts"], n_groups=A[n]["n_groups"],
                                      summary={m: A[n]["summary"][m] for m in BASELINES + [ENS, FULL]},
                                      tests={k: {f: A[n]["tests"][k][f] for f in ("full_vs_baselines", "full_vs_baselines_and_ensemble",
                                                                                 "ensemble_vs_single")} for k in TESTED},
                                      gain_decomposition=A[n]["gain_decomposition"]) for n in SUBSETS})
    (OUT / "main_results.json").write_text(json.dumps(main_json, indent=1))
    rows = []
    for n in SUBSETS:
        rows += rows_for(n, A[n], BASELINES + [ENS, FULL], "full_vs_baselines_and_ensemble", "Holm over 8 (7 baselines + ensemble)")
    pd.DataFrame(rows).to_csv(OUT / "main_results.csv", index=False)
    rows = []
    for n in SUBSETS:
        rows += rows_for(n, A[n], [SINGLE, ENS] + CONSTRUCTIVE + MULTI + [FULL], "full_vs_ensemble_and_ablations",
                         "Holm over 9 (ensemble + constructive + multi-module)")
        rows += rows_for(n, A[n], LOO, "full_vs_single_module_ablations", "Holm over 6 (single-component ablations)")
    pd.DataFrame(rows).to_csv(OUT / "ablation_results.csv", index=False)
    (OUT / "ablation_results.json").write_text(json.dumps(
        {n: dict(n_excerpts=v["n_excerpts"], n_groups=v["n_groups"], summary=v["summary"], tests=v["tests"],
                 gain_decomposition=v["gain_decomposition"]) for n, v in A.items()}, indent=1))
    if a.skip_decode_check:
        rep["decode_check"] = dict(skipped=True, passed=True)
    else:
        log("4/4 live decoding check")
        rep["decode_check"] = decode_check(EX, P, a.device)
    rep["runtime_s"] = round(time.time() - t0, 1)
    rep["all_passed"] = bool(all(rep[k]["passed"] for k in ("integrity", "per_excerpt_recompute", "headline_values", "decode_check")))
    (OUT / "verification_report.json").write_text(json.dumps(rep, indent=1))
    log(f"done in {rep['runtime_s']} s; all checks passed: {rep['all_passed']}")
    for n in SUBSETS:
        S = A[n]["summary"]
        log(f"{n}: DECGD_full PI-Macro-F1 {S[FULL]['pi_macro_f1']['mean']:.3f}, PI-Acc {S[FULL]['pi_acc']['mean']:.3f}; "
            f"B3 ensemble {S[ENS]['pi_macro_f1']['mean']:.3f}; B3 single {S[SINGLE]['pi_macro_f1']['mean']:.3f}")
    sys.exit(0 if rep["all_passed"] else 1)


if __name__ == "__main__":
    main()
