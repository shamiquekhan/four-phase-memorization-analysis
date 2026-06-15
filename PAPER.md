# Paper-to-Code Mapping

**Working title:** Structural Fingerprints of Label Memorization in Shallow Neural Networks

Maps claims and results to specific source files and functions.
Empirical values reflect hidden_dim=16 architecture (784→16→10), 10 seeds, 95% CI.

## Abstract

| Claim | Empirical Value | Code Location |
|-------|----------------|--------------|
| Class separability (Fisher discriminant ratio) | FDR=0.862 (h=16), decreases to 0.387 (h=1024) | `src/utils/metrics.py:compute_fdr()`, `src/scaling/analyze_scaling.py` |
| Monosemantic neuron fraction (ρ>0.5) | 26.9% [20.2%, 33.6%] | `src/utils/metrics.py:compute_monosemanticity()` |
| Mean circuit size | 6.6 [6.1, 7.1] neurons/class | `src/utils/metrics.py:compute_circuit_sparsity()` |
| Clean test accuracy | 95.31% [94.88%, 95.74%] | `src/training/train_clean.py` |
| Corrupted test accuracy (20% noise) | 93.71% [93.42%, 94.00%] | `src/training/train_corrupted.py` |
| ROME recovery (targeted corruption) | 10–22% (varies by config, 5 seeds) | `src/analysis/multiclass_rome.py:run_rome_experiment()` |
| ROME random baseline (signal ratio) | ∞ (random recovers 0%) | `src/analysis/multiclass_rome.py:run_rome_with_random_baseline()` |
| Gradient alignment at convergence | +0.9944 [+0.9926, +0.9961] (near-perfect, noise-dominated) | `src/analysis/phase3_influence.py:compute_gradient_alignment()` |
| Multi-layer ROME (sequential + joint) | 22–40% recovery (vs 10–22% single) | `src/analysis/multilayer_rome.py:run_multilayer_rome_experiment()` |
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
| Scaling metrics (FDR, DB index, mono, sparsity vs width) | `src/scaling/analyze_scaling.py` | `run_scaling_analysis()` |
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

## Related Work

**Zhang et al. (2017) — "Understanding deep learning requires rethinking generalization"**

Zhang et al. demonstrated that deep networks can memorize randomly labeled training data while achieving near-zero training loss, challenging the traditional view that generalization requires inductive biases. They established that networks are expressive enough to memorize any labeling of any dataset.

*Our delta*: Zhang et al. show that memorization is possible; we show where and how it is encoded structurally. Our CKA analysis localizes the memorization signature to the nonlinear activation layer, and ROME provides a causal intervention that confirms the localization — moving from "networks can memorize" to "here is the geometric fingerprint."

**Arpit et al. (2017) — "A Closer Look at Memorization in Deep Networks"**

Arpit et al. showed that networks learn simple patterns first and memorize harder or noise-corrupted examples later, consistent with an implicit curriculum. They introduced the concept of "memorization fraction" and showed it correlates with training loss.

*Our delta*: We replace the loss-quantile definition of memorization (which is partially circular — memorized samples have high loss because they are memorized) with a ground-truth definition using corruption indices saved during training. This eliminates circularity from the memorization metric and allows unbiased estimation of the loss gap between memorized and genuine samples. We also extend the analysis to causal interventions (ROME) that Arpit et al. do not consider.

**Koh & Liang (2017) — "Understanding Black-box Predictions via Influence Functions"**

Koh & Liang adapted influence functions from robust statistics to machine learning, enabling per-sample attribution of model predictions to training points. They showed that influence functions can identify mislabeled training data.

*Our delta*: We use influence functions as one of four analysis phases, situating them within a broader geometric and causal framework. Critically, we show that influence functions alone (Phase 3) do not localize memorization to specific weight matrices — for causal localization, Phase 4 (ROME) is required. The two methods are complementary: influence functions identify *which samples* are memorized; ROME identifies *where in the network* the memorization is encoded.

**Meng et al. (2022) — "Locating and Editing Factual Associations in GPT"**

Meng et al. introduced ROME for causal localization and targeted editing of factual associations in large language models. They showed that factual knowledge in GPT-style models is localized to a small set of MLP layers, and that rank-one weight edits can modify specific facts without disrupting other knowledge.

*Our delta*: We transfer ROME from autoregressive language models (where it localizes factual knowledge) to classification networks (where it localizes memorized label associations). The key insight is that the ROME formula — Δ = outer(v - W·u, u) / (u·u) — is architecture-agnostic. We show that ROME can reliably detect memorized associations in shallow classifiers (10–22% recovery, signal ratio 10–20× over random baseline), establishing a new use case for causal weight editing in memorization analysis.

**Kornblith et al. (2019) — "Similarity of Neural Network Representations Revisited"**

Kornblith et al. established centered kernel alignment (CKA) as a reliable measure of representation similarity across layers and models. They showed that CKA is invariant to orthogonal transformations and isotropic scaling.

*Our delta*: We use CKA not as a cross-network comparison tool but as an intra-network probe of memorization geometry. The key finding — that the ReLU activation (fc1_pre → fc1_post) shows a CKA drop of Δ = −0.160 under 20% label noise while flanking linear layers show Δ ≈ 0 — suggests a new diagnostic use of CKA as a layer-level memorization detector. This complements the original cross-network comparison use case.

**Elhage et al. (2022) — "Toy Models of Superposition"**

Elhage et al. introduced the monosemanticity framework: the hypothesis that neurons in large networks encode multiple features simultaneously (polysemanticity), while neurons in small networks or low-rank approximations tend toward single-feature specialization (monosemanticity).

*Our delta*: We show that monosemanticity fraction decreases monotonically with hidden dimension (0.269 at h=16 → 0.075 at h=1024), consistent with the superposition hypothesis. Critically, we link monosemanticity to memorization capacity: at h=16, the 26.9% monosemantic neuron fraction corresponds to a regime where individual neuron specialization is observable and ROME edits are most localized. This provides empirical support for the superposition hypothesis in the context of label memorization rather than feature superposition.

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
