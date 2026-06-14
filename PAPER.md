# Paper-to-Code Mapping

**Working title:** Structural Fingerprints of Label Memorization in Shallow Neural Networks

Maps claims and results to specific source files and functions.
Empirical values reflect hidden_dim=16 architecture (784→16→10), 10 seeds, 95% CI.

## Abstract

| Claim | Empirical Value | Code Location |
|-------|----------------|--------------|
| Class separability (Davies-Bouldin, lower is better) | DB=0.693 [0.674, 0.712] | `src/utils/metrics.py:compute_davies_bouldin()`, `src/analysis/phase2_representation.py` |
| Monosemantic neuron fraction (ρ>0.5) | 26.9% [20.2%, 33.6%] | `src/utils/metrics.py:compute_monosemanticity()` |
| Mean circuit size | 6.6 [6.1, 7.1] neurons/class | `src/utils/metrics.py:compute_circuit_sparsity()` |
| Clean test accuracy | 95.31% [94.88%, 95.74%] | `src/training/train_clean.py` |
| Corrupted test accuracy (20% noise) | 93.71% [93.42%, 94.00%] | `src/training/train_corrupted.py` |
| ROME recovery (targeted corruption) | 10–20% (varies by config, 5 seeds) | `src/analysis/multiclass_rome.py:run_rome_experiment()` |
| ROME noise sweep (10%/20%/40%) | Ratio scales monotonically: 3.37×→4.34×→5.98× | `scripts/run_rome_noise_sweep.py`, `scripts/run_rome_fast.py` |
| Baseline: spectral norm vs ROME | ROME (4.34×) > spectral (1.86×) | `scripts/baseline_comparison.py` |
| Baseline: linear probe AUC | 0.514 (barely above random) | `scripts/baseline_comparison.py` |

## Phase 1: Geometric Analysis

| Result | File | Function |
|--------|------|----------|
| Davies-Bouldin index (replaces σ) | `src/utils/metrics.py` | `compute_davies_bouldin()` |
| Weight norms (Frobenius + spectral) | `src/analysis/phase1_basic.py` | `compute_weight_norms()` |
| Gradient norms | `src/analysis/phase1_basic.py` | `compute_gradient_norms()` |
| Epoch-wise accuracy/loss curves | `src/analysis/phase1_basic.py` | Training loop in `main()` |

## Phase 2: Representation Analysis

| Result | File | Function |
|--------|------|----------|
| CKA similarity between layers | `src/analysis/phase2_representation.py` | `linear_cka()` |
| Activation statistics (sparsity, mean, std) | `src/analysis/phase2_representation.py` | `analyze_representations()` |
| PCA variance explained | `src/analysis/phase2_representation.py` | `compute_pca()` |
| t-SNE manifold progression | `src/analysis/visualizations.py` | `figure3_tsne_manifold()` |

## Phase 3: Influence & Memorization

| Result | File | Function |
|--------|------|----------|
| Per-sample memorization | `src/analysis/phase3_influence.py` | `compute_memorization_score()` |
| Influence via conjugate gradient | `src/analysis/phase3_influence.py` | `conjugate_gradient()`, `compute_influence()` |
| Circuit sparsity | `src/utils/metrics.py` | `compute_circuit_sparsity()` |

## Phase 4: Causal Interventions (ROME)

| Result | File | Function |
|--------|------|----------|
| ROME edit computation (standard formula) | `src/analysis/phase4_rome.py` | `compute_rome_edit()` |
| Multi-class ROME validation | `src/analysis/multiclass_rome.py` | `run_rome_experiment()` |
| Rank ablation study | `src/analysis/rank_ablation.py` | `run_rank_ablation()` |
| Recovery across corruption configs | `src/analysis/multiclass_rome.py` | 4 configs: 7→1, 1→7, 5→6, 0→8 |
| ROME formula: `Δ = outer(v - W·u, u) / (u·u)` | `src/analysis/multiclass_rome.py` | `apply_rome()` |

## Scaling Experiment

| Result | File | Function |
|--------|------|----------|
| Train models across widths (16–256) | `src/scaling/train_scaling.py` | `main()` |
| Scaling metrics (DB index, mono, sparsity vs width) | `src/scaling/analyze_scaling.py` | `run_scaling_analysis()` |
| Power-law fit | `src/scaling/analyze_scaling.py` | `power_law()`, `curve_fit` |

## Statistical Validation

| Result | File | Function |
|--------|------|----------|
| 95% CI across 10 seeds | `src/utils/stats.py` | `compute_ci()` |
| Multi-seed experiment runner | `src/utils/stats.py` | `run_with_seeds()` |
| CI aggregation | `src/utils/stats.py` | `aggregate_results()` |
| Paired t-test | `src/utils/stats.py` | `paired_t_test()` |

## Figures

| Figure | Description | Function |
|--------|-------------|----------|
| 1 | Four-phase framework diagram | `figure1_framework()` |
| 2 | ROME pipeline diagram | `figure2_rome_pipeline()` |
| 3 | t-SNE manifold progression across layers | `figure3_tsne_manifold()` |
| 4 | Hidden neuron weight vectors (28×28) | `figure4_neuron_weights()` |
| 5 | Neuron-class correlation heatmap | `figure5_correlation_heatmap()` |
| 6 | Circuit map (critical neuron-class connections) | `figure6_circuit_map()` |
| 7 | ROME accuracy recovery bar chart | `figure7_rome_recovery()` |
| 8 | Scaling analysis (DB index, mono, circuit, sparsity vs width) | `figure8_scaling_analysis()` |

## Reproducibility

| Item | Location |
|------|----------|
| Full pipeline | `reproduce_all.py` |
| Configuration | `configs/experiment_config.yaml` |
| Seeds | `src/utils/stats.py:SEEDS` (10 seeds: 42, 123, ..., 9999) |
| Environment | `environment.yml`, `requirements.txt` |
| Tests | `tests/test_metrics.py` |
| Checkpoints | `outputs/{clean,corrupted,targeted_corrupted,scaling}/seed_*/` |
| Figure outputs | `outputs/figures/` (PDF + PNG) |
| Noise sweep data | `outputs/analysis/noise_sweep_rome.json`, `outputs/analysis/rome_08_rank_ablation.json` |
| Baseline comparison data | `outputs/analysis/baseline_comparison.json` |
