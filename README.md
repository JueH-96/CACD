# DECGD: held-out same-timbre guitar-duet evaluation

One-command reproduction of the held-out evaluation of DECGD, a decoder that assigns each note of a same-timbre rhythm/lead guitar duet to one of two streams. Three scoring components are combined by a joint decoder:

1. **Swap-contrastive symbolic prior.** A causal note-set language model pre-trained on Lakh MIDI scores each stream under its own history minus the score under the swapped histories (Δ_swap).
2. **Capacity-calibrated GNN-Transformer ensemble.** Six B3 networks (five dev-fold, one full-dev) vote with permutation-aligned node probabilities and pair logits; an unsupervised per-excerpt minor-stream share ρ̂ drives polarity alignment, quantile logit re-centring and within-cluster sharpening.
3. **Acoustic posteriors and fretboard playability.** B2 acoustic-symbolic HMM posteriors plus a six-string, 24-fret playability model (EADGBE, one note per string, fret-span limit, hand-shift cost).

The decoder runs a warm-started beam search over onset clusters with an adaptive capacity-equilibrium term, a multi-proposal pool and block-coordinate refinement.

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh                          # about 2 min; exit code 0 = all checks passed
bash run_all.sh --skip-decode-check      # statistics only
bash run_all.sh --device cpu             # force CPU for live decoding
PYTHON=/path/to/python bash run_all.sh   # choose the interpreter
```

## Checks performed by `run_all.sh`

1. **Integrity.** SHA-256 of every bundled file against `integrity_lock.json`.
2. **Metrics.** Per-excerpt metrics for 23 configurations on the 69 guitar duets, recomputed from cached predictions and ground-truth notes and compared with `data/reference_per_excerpt_metrics.csv` and `data/expected_headline_values.json` (tolerance 1e-12): PI-Macro-F1, PI-Acc (best global label permutation), VSER (voice-swap error rate), Dense Acc (clusters of ≥ 3 notes) and Imb Err (|Imb_pred − Imb_gt|, Imb = |N₀ − N₁| / N).
3. **Statistics.** For all guitar duets, GuitarSet and Lakh guitar: two-sided paired Wilcoxon signed-rank tests (excerpt and piece-group level), Holm correction within each family, matched-pairs rank-biserial r, and 10,000-resample piece-group bootstrap 95% CIs (seed 2026).
4. **Live decoding.** B3 single, B3 ensemble, B2 posterior, a prior-free DECGD configuration and full DECGD are re-run on three excerpts (two shortest Lakh, shortest GuitarSet); labels must match the cached predictions up to relabelling.

## Benchmark

| Subset | Excerpts | Piece groups | Audio |
|---|---|---|---|
| All guitar duets | 69 | 48 | — |
| GuitarSet guitar | 14 | 7 | real comp and solo recordings mixed into duets |
| Lakh guitar | 55 | 41 | two Lakh MIDI guitar stems rendered with one guitar timbre |

Piece-group identifiers are anonymised (`g001`, …); excerpts of one piece share a group, and group-level tests and bootstrap resampling operate on groups.

## Configurations (`method` column)

| Name | Description |
|---|---|
| `B1_viterbi_dp` | pitch/proximity Viterbi dynamic programming |
| `B2_acoustic_symbolic_hmm` | spectral clustering + pitch-continuity HMM on acoustic features |
| `B3_single_model` | single supervised GNN-Transformer, threshold 0.5 |
| `B4_naive_lm_prior` | decoding with the uncontrasted prior likelihood |
| `B3_ensemble_only` | six-network B3 ensemble with chunk polarity alignment, threshold 0.5 |
| `decgd_classic` | generative prior + playability + refinement |
| `decgd_hybrid_full`, `decgd_hybrid_all` | hybrid decoders using the single B3 network |
| `ens_decoder_only` | ensemble potentials in the joint decoder, other components off |
| `ens_plus_*` | ensemble + one or two components |
| `DECGD_wo_*` | full model with one component or component family removed |
| `DECGD_full` | full model (configuration selected by nested cross-validation on dev) |

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `main_results.csv/.json` | per-subset means, SD, CIs, secondary metrics, full-model differences with raw and Holm-corrected p-values, gain decomposition |
| `ablation_results.csv/.json` | ensemble, component-addition and component-removal configurations with Holm-corrected tests |
| `per_excerpt_metrics.csv` | recomputed per-excerpt metrics |
| `verification_report.json` | results of all checks, environment versions, runtime |
| `run_log.txt` | console log |

Expected PI-Macro-F1 (mean over excerpts):

| Configuration | All guitar (69) | GuitarSet (14) | Lakh guitar (55) |
|---|---|---|---|
| `DECGD_full` | 0.795 | 0.872 | 0.775 |
| `B3_ensemble_only` | 0.757 | 0.850 | 0.733 |
| `decgd_hybrid_all` | 0.753 | 0.859 | 0.726 |
| `B3_single_model` | 0.740 | 0.836 | 0.716 |

## Layout

```
run_all.sh, reproduce.py   entry points
src/                       decoder, baselines, prior, metrics and statistics modules
data/                      benchmark index, ground-truth notes, cached predictions, reference metrics, features
checkpoints/               six B3 networks and the priors used for live decoding
configs/                   live-decoding configurations
integrity_lock.json        SHA-256 of all bundled files
```

## Scope

The package covers evaluation from cached predictions plus live re-decoding of three excerpts; training code and development data are not included. The 69 guitar duets are the guitar subset of a 120-excerpt held-out split.
