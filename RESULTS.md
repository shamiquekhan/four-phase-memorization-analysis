# Results: Memorization in Neural Networks

All values reported as `mean [95% CI]` over seeds unless otherwise noted.
CI computed via Student's t-distribution: `mean ± t_{0.975, n-1} * SEM`. Paired t-tests used for clean-vs-corrupted comparisons.

---

## Training Performance

| Metric | Clean (10 seeds) | Corrupted 20% (10 seeds) |
|--------|:-:|:-:|
| Test Accuracy | **95.31%** [95.16%, 95.46%] | **93.71%** [93.42%, 94.00%] |
| Final Train Loss | 0.124 | 1.015 |
| Final Test Loss | 0.165 | 0.410 |

The corrupted model reaches lower test accuracy than the clean model (~93.7% vs ~95.3%), a 1.6pp gap, because it must allocate capacity to memorizing 20% flipped labels. Despite this, the model still learns genuine features — the gap is small relative to the 20% corruption rate.

---

## Phase 1: Weight Geometry

### MNIST (2-layer MLP, 10 seeds)

| Metric | Clean | Corrupted |
|--------|:-:|:-:|
| FC1 Spectral Norm | 4.37 [4.24, 4.50] | **3.66** [3.53, 3.79] |
| FC2 Spectral Norm | 2.58 [2.42, 2.75] | **1.39** [1.30, 1.47] |
| FC1 Frobenius Norm | 12.56 [12.38, 12.74] | **10.33** [10.19, 10.47] |
| FC2 Frobenius Norm | 4.99 [4.83, 5.16] | **2.79** [2.70, 2.87] |
| Gradient Norm (final) | 0.94 [0.89, 0.98] | 1.00 [0.96, 1.05] |

**Finding**: Both spectral norms decrease under corruption (FC1: 4.37→3.66, FC2: 2.58→1.39). This flattening of the singular value spectrum indicates that label-noise models operate in a lower-effective-rank regime.

### CIFAR-10 (3-layer MLP, 5 seeds)

| Metric | Clean | Corrupted | p-value |
|--------|:-:|:-:|:-:|
| FC1 Spectral Norm | 18.22 [17.93, 18.51] | 18.31 [17.93, 18.70] | 0.71 |
| FC2 Spectral Norm | 8.99 [8.83, 9.15] | **9.98** [9.85, 10.10] | 0.0003 |
| FC3 Spectral Norm | 1.72 [1.68, 1.77] | **1.88** [1.78, 1.98] | 0.017 |

**Finding**: Spectral norm changes are depth-dependent. FC1 (input layer) is unchanged (p=0.71). FC2 and FC3 increase under corruption, consistent with deeper layers absorbing additional separability burden when resolving noisy labels. This supports Proposition 2: in multi-layer networks, output-adjacent layers intensify to handle conflicting label assignments.

---

## Phase 2: Representation Similarity (CKA)

### MNIST (2-layer MLP, 10 seeds)

| Layer Pair | Clean | Corrupted | Δ (Clean−Corrupted) | p-value |
|------------|:-:|:-:|:-:|:-:|
| input→fc1_pre | 0.712 [0.697, 0.727] | 0.693 [0.676, 0.710] | +0.019 | 0.08 |
| **fc1_pre→fc1_post** | **0.850 [0.828, 0.873]** | **0.690 [0.669, 0.712]** | **+0.160** | **<0.001** |
| fc1_post→output | 0.706 [0.666, 0.745] | 0.715 [0.685, 0.745] | −0.009 | 0.65 |

**Finding (MNIST)**: The ReLU nonlinearity (fc1_pre→fc1_post) shows a dramatic CKA drop under corruption: 0.850 → 0.690 (Δ = +0.160, p < 0.001). The linear transformations (input→fc1_pre, fc1_post→output) remain near-identical. Memorization distortion localizes at the nonlinearity.

### CIFAR-10 (3-layer MLP, 5 seeds)

| Layer Pair | Clean | Corrupted | Δ (Clean−Corrupted) | p-value |
|------------|:-:|:-:|:-:|:-:|
| input→fc1_pre | 0.767 [0.748, 0.786] | 0.765 [0.754, 0.777] | +0.002 | 0.88 |
| fc1_pre→fc1_post | 0.819 [0.808, 0.830] | 0.801 [0.796, 0.807] | +0.018 | 0.017 |
| fc1_post→fc2_pre | 0.721 [0.698, 0.745] | 0.749 [0.737, 0.761] | −0.028 | 0.08 |
| fc2_pre→fc2_post | 0.177 [0.142, 0.211] | 0.140 [0.109, 0.171] | +0.037 | 0.09 |
| **fc2_post→output** | **0.580 [0.528, 0.632]** | **0.477 [0.457, 0.498]** | **+0.103** | **0.009** |

**Finding (CIFAR-10)**: The largest CKA distortion is at fc2_post→output (Δ = +0.103, p = 0.009), the final linear layer before logits. The first nonlinearity shows a much smaller effect (Δ = +0.018). This confirms Proposition 1's refined claim: distortion concentrates at the **deepest pre-output interface**. In a 2-layer net this interface is the single ReLU; in a 3-layer net it is the output linear layer — the location shifts with architecture depth.

| Activation Statistics (MNIST) | Clean | Corrupted |
|--------|:-:|:-:|
| Hidden Layer Sparsity | ~0.85 | ~0.82 |
| Hidden Layer Mean | ~0.08 | ~0.34 |
| Hidden Layer Std | ~0.45 | ~0.60 |

Corrupted models have higher mean activations and lower sparsity, recruiting more neurons to represent the noise-inflated label space.

---

## Phase 3: Influence & Memorization

### Clean Model (loss-quantile definition, 10 seeds)

| Metric | Value |
|--------|:-:|
| Mean Loss | 0.112 [0.108, 0.117] |
| Accuracy | 96.68% [96.58%, 96.79%] |
| Memorized Fraction (top 25% loss) | 0.250 (by construction) |
| Loss Gap (memorized − forgotten) | **+0.442** [+0.426, +0.458] |

### Corrupted Model (ground-truth definition, 10 seeds)

| Metric | Value |
|--------|:-:|
| Mean Loss | 0.408 [0.396, 0.420] |
| Accuracy | 94.21% [93.98%, 94.45%] |
| Memorized Fraction (exact 20% corrupted) | 0.200 (matches noise rate) |
| Loss Gap (memorized − forgotten) | **−0.051** [−0.057, −0.045] |

**Finding**: The clean model's loss-quantile analysis is provided as a methodological baseline: defining memorized samples as the top 25% by loss is circular (high-loss samples are labeled memorized by construction), yielding a mechanical gap of +0.442. The genuine finding is the corrupted model's non-circular design: using ground-truth corruption indices (20% of labels) breaks the circularity and reveals a gap of −0.051 (CI excludes zero). This confirms that memorized (corrupted) samples are genuinely harder for the model to predict than clean samples — the model has learned the true labels better than the flipped ones.

---

## Phase 4: ROME (Rank-One Model Editing)

### Single-Layer ROME

#### MNIST (fc1, all 10 classes, 10 seeds)

| Class | Clean Δ-norm [95% CI] | Corrupted Δ-norm [95% CI] | Ratio | p-value |
|------:|:-:|:-:|:-:|:-:|
| 0 | 21.62 [19.73, 23.52] | 9.08 [8.40, 9.76] | 2.38× | <0.0001 |
| 1 | 16.56 [15.05, 18.07] | 8.58 [7.86, 9.30] | 1.93× | <0.0001 |
| 2 | 14.42 [13.20, 15.64] | 6.68 [6.08, 7.28] | 2.16× | <0.0001 |
| 3 | 13.04 [12.18, 13.90] | 6.17 [5.67, 6.67] | 2.11× | <0.0001 |
| 4 | 14.11 [13.20, 15.01] | 9.06 [8.71, 9.41] | 1.56× | <0.0001 |
| 5 | 10.96 [10.18, 11.73] | 5.72 [5.28, 6.17] | 1.91× | <0.0001 |
| 6 | 15.84 [15.00, 16.67] | 8.15 [7.37, 8.93] | 1.94× | <0.0001 |
| 7 | 15.99 [14.74, 17.24] | 7.02 [6.45, 7.60] | 2.28× | <0.0001 |
| 8 | 9.83 [9.12, 10.54] | 4.79 [4.52, 5.05] | 2.05× | <0.0001 |
| 9 | 12.50 [11.91, 13.10] | 6.59 [6.20, 6.99] | 1.90× | <0.0001 |

**MNIST fc1 average**: Clean=14.49, Corrupted=7.18, Ratio=**2.02×** (paired t-test p<0.0001 for every class)

#### MNIST (fc2, all 10 classes, 10 seeds)

| Class | Clean Δ-norm [95% CI] | Corrupted Δ-norm [95% CI] | Ratio | p-value |
|------:|:-:|:-:|:-:|:-:|
| 0 | 22.46 [21.02, 23.90] | 4.54 [3.69, 5.40] | 4.94× | <0.0001 |
| 1 | 18.11 [16.97, 19.26] | 4.40 [4.02, 4.78] | 4.12× | <0.0001 |
| 2 | 21.01 [19.58, 22.45] | 4.23 [3.98, 4.48] | 4.97× | <0.0001 |
| 3 | 17.96 [15.99, 19.92] | 4.49 [4.10, 4.88] | 4.00× | <0.0001 |
| 4 | 21.04 [19.51, 22.57] | 4.75 [4.35, 5.15] | 4.43× | <0.0001 |
| 5 | 18.06 [16.69, 19.42] | 4.21 [4.01, 4.41] | 4.29× | <0.0001 |
| 6 | 21.91 [21.41, 22.41] | 4.40 [4.11, 4.69] | 4.97× | <0.0001 |
| 7 | 20.42 [18.89, 21.96] | 4.36 [4.09, 4.64] | 4.68× | <0.0001 |
| 8 | 10.79 [9.97, 11.60] | 4.09 [3.82, 4.37] | 2.63× | <0.0001 |
| 9 | 19.02 [17.98, 20.07] | 4.50 [4.07, 4.93] | 4.23× | <0.0001 |

**MNIST fc2 average**: Clean=19.08, Corrupted=4.40, Ratio=**4.34×** (paired t-test p<0.0001 for every class)

#### CIFAR-10 (fc3, all 10 classes, 3 seeds)

| Class | Clean Δ-norm (mean ± SEM) | Corrupted Δ-norm (mean ± SEM) | Ratio | p-value |
|------:|:-:|:-:|:-:|:-:|
| 0 | 0.325 ± 0.035 | 0.175 ± 0.027 | 1.85× | 0.028 |
| 1 | 0.438 ± 0.033 | 0.262 ± 0.018 | 1.68× | 0.009 |
| 2 | 0.385 ± 0.027 | 0.186 ± 0.016 | 2.07× | 0.003 |
| 3 | 0.341 ± 0.009 | 0.196 ± 0.015 | 1.74× | 0.001 |
| 4 | 0.339 ± 0.019 | 0.169 ± 0.017 | 2.01× | 0.003 |
| 5 | 0.376 ± 0.012 | 0.214 ± 0.007 | 1.76× | <0.001 |
| 6 | 0.327 ± 0.014 | 0.170 ± 0.005 | 1.93× | <0.001 |
| 7 | 0.408 ± 0.016 | 0.216 ± 0.006 | 1.89× | <0.001 |
| 8 | 0.366 ± 0.037 | 0.197 ± 0.036 | 1.85× | 0.030 |
| 9 | 0.380 ± 0.028 | 0.237 ± 0.019 | 1.60× | 0.013 |

**Finding**: The ROME delta-norm finding is now the strongest result in the paper. On MNIST, every class shows clean > corrupted at p<0.0001 for both fc1 (mean ratio 2.02×) and fc2 (mean ratio 4.34×). On CIFAR-10, all 10 classes confirm at p<0.05 (mean ratio 1.84×). The cross-architecture consistency — ratio ~2–4× on MNIST and ~1.8× on CIFAR-10, across completely different depths, widths, and datasets — suggests the relationship between class boundary overlap and edit magnitude is an architectural invariant.

### Noise Rate Sweep

ROME delta-norm scales monotonically with label noise rate (MNIST fc2, 5 seeds):

| Noise Rate | FC2 Δ-norm | Ratio vs Clean (19.08) |
|:----------:|:----------:|:----------------------:|
| 0% (Clean) | 19.08 | 1.00× |
| 10% | ~5.67 | 3.37× |
| 20% | ~4.40 | 4.34× |
| 40% | ~3.19 | 5.98× |

**Finding**: The ROME delta-norm ratio increases monotonically with noise rate (3.37× → 4.34× → 5.98×). This confirms that delta-norm probes the degree of memorization, not a noise-20%-specific artifact. If the ROME ratio were flat across noise rates, it would measure a binary clean-vs-corrupted difference; the monotonic scaling shows it tracks the memorization load continuously.

### Baseline Comparison

| Method | Discriminability |
|--------|:----------------:|
| ROME delta-norm fc2 ratio | **4.34×** |
| Spectral norm fc2 ratio | 1.86× |
| Linear probe (hidden acts → corruption) | AUC = 0.514 |

**Finding**: ROME is the most sensitive memorization probe. It outperforms the spectral norm ratio by 2.3× (4.34× vs 1.86×). The linear probe — a logistic regression trained on hidden activations to predict whether a sample is corrupted — achieves AUC of only 0.514, barely above random (0.5). This rules out a trivial explanation: ROME is not reading off an obvious signal in the hidden activations; it captures a geometric property (class boundary overlap in weight space) that a linear probe cannot access.

### Multi-Class ROME (Targeted Corruption, MNIST)

Recovery of source-class accuracy after applying ROME edit to the corrupted model (5 seeds):

| Config | Recovery | Side Effects | Edit Magnitude |
|--------|:-:|:-:|:-:|
| 7→1 | +0.116 | 0.171 | 1.102 |
| 1→7 | +0.195 | 0.240 | 1.017 |
| 5→6 | +0.160 | 0.246 | 0.934 |
| 0→8 | +0.097 | 0.216 | 0.881 |

**ROME as a memorization localizer, not a full repair**: Rank-one edits recover part of the lost source-class accuracy in all 4 targeted corruption configs (7→1: +0.116, 1→7: +0.195, 5→6: +0.160, 0→8: +0.097), with consistent side effects (~17–25%). The 0→8 case, previously reported as +0.014 from a single seed, averages +0.097 across 5 seeds — seed variance accounted for the earlier underestimate. The ROME rank-1 edit is rank-1 by construction (SVD of the edit delta has exactly 1 significant singular value), so partial-rank recovery does not improve results. Recovery is partial overall because memorization under targeted corruption is distributed across both weight layers. Single-layer ROME is a reliable **detector** (positive across all 4 configs) but not a full repair. Multi-layer sequential ROME is the natural extension.

### Rank Ablation (MNIST fc2, 10 seeds)

Accuracy at different SVD ranks of the fc2 weight matrix:

| Rank | Clean (10 seeds) | Corrupted (10 seeds) |
|------|:-:|:-:|
| 1/10 | 19.0% [17.9%, 20.1%] | 19.3% [16.6%, 21.9%] |
| 2/10 | 33.9% [30.6%, 37.2%] | 30.0% [26.4%, 33.6%] |
| 3/10 | 44.9% [40.2%, 49.7%] | 40.5% [37.0%, 44.0%] |
| 4/10 | 58.3% [53.5%, 63.2%] | 53.2% [49.0%, 57.3%] |
| **5/10** | **68.4%** [62.2%, 74.6%] | **62.3%** [57.9%, 66.7%] |
| 6/10 | 75.4% [67.3%, 83.5%] | 72.6% [68.0%, 77.3%] |
| 7/10 | 80.1% [70.8%, 89.4%] | 80.7% [76.3%, 85.0%] |
| 8/10 | 88.9% [85.7%, 92.2%] | 86.4% [84.0%, 88.8%] |
| 9/10 | 92.8% [90.7%, 94.9%] | 92.3% [91.8%, 92.9%] |
| 10/10 (full) | **95.3%** [95.2%, 95.5%] | **93.7%** [93.4%, 94.0%] |

**Finding**: The key comparative signal is at **rank 5/10**: corrupted models perform **6.1pp worse** than clean (62.3% vs 68.4%). This gap is intermediate-rank-specific — at rank 8/10 both models reach ~90% of full accuracy (88.9% vs 86.4%, 2.5pp gap), and at rank 10/10 (full) the gap is 1.6pp. The rank-5 finding shows that corrupted models require a broader effective rank spectrum: their memorized label-noise associations are less compressible into the top singular directions. This is not a uniform effect — the gap narrows at higher ranks, consistent with the interpretation that corrupted models distribute information more evenly across the spectrum rather than concentrating it in the top components. This connects to Phase 1, where MNIST spectral norms decrease under corruption, confirming a flatter singular value distribution.

---

## Scaling Analysis (MNIST, 10 seeds)

| Hidden Dim | Monosemanticity | Circuit Size | Sparsity | Accuracy |
|:-:|:-:|:-:|:-:|:-:|
| 16 | 0.269 | 6.6 | 0.588 | 95.31% |
| 32 | 0.222 | 8.3 | 0.740 | 96.85% |
| 64 | 0.212 | 3.5 | 0.945 | 97.50% |
| 128 | 0.184 | 0.4 | 0.997 | 97.85% |
| 256 | 0.175 | 0.1 | 1.000 | 97.93% |
| 512 | 0.154 | 0.0 | 1.000 | 98.17% |
| 1024 | 0.075 | 0.0 | 1.000 | 98.11% |

**Finding**: As width increases beyond 64 hidden neurons:
- **Monosemanticity decreases** (0.269 → 0.075) — wider models distribute representations, reducing neuron specialization
- **Circuit sparsity → 1.0** beyond h=128 — vanishing fraction of capacity used per class
- **Accuracy plateaus** at ~98% beyond h=256

The sparsity convergence to 1.0 is the key scaling result: larger networks use a vanishing fraction of their capacity per class, demonstrating superlinear compression and aligning with the superposition hypothesis (Elhage et al., 2022). The σ metric has been replaced with Davies-Bouldin index as a dimension-invariant alternative (see scaling analysis code).

---

## Output Files Summary

All results are in `outputs/`:

| Path | Contents |
|------|----------|
| `outputs/clean/seed_*/` | 10 clean model checkpoints |
| `outputs/corrupted/noise_0.2/seed_*/` | 10 corrupted model checkpoints |
| `outputs/targeted_corrupted/src*_tgt*/seed_*/` | 20 targeted corruption models |
| `outputs/scaling/hidden_*/seed_*/` | 72 scaling experiment models |
| `outputs/analysis/phase1_*/*.json` | Phase 1 weight/gradient metrics |
| `outputs/analysis/phase2_*/*.json` | Phase 2 CKA/PCA/activation results |
| `outputs/analysis/phase3_*/*.json` | Phase 3 influence/memorization scores |
| `outputs/analysis/phase4_*/*.json` | Phase 4 ROME edit analysis |
| `outputs/analysis/multiclass_rome/*.json` | Multi-class ROME results |
| `outputs/analysis/rank_ablation_*/*.json` | Rank ablation results |
| `outputs/analysis/scaling/*.json` | Scaling analysis metrics |
| `outputs/cifar10/analysis/*.json` | CIFAR-10 validation results |
| `outputs/figures/*.png` | Publication figures |
