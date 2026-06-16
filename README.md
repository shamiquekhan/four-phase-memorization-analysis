# Structural Fingerprints of Label Memorization in Shallow Neural Networks

A systematic 4-phase analysis of how label memorization leaves structural fingerprints in shallow ReLU networks — spanning CKA representation drift, spectral geometry, circuit sparsity, influence functions, and rank-one model editing (ROME). Primary experiments on MNIST (784→16→10), validated on CIFAR-10 (3-layer MLP), with width scaling from 16 to 1024 hidden units.

## Key Results (MNIST, 10 seeds, 95% CI)

| Metric | Clean | Corrupted (20% noise) |
|--------|:-----:|:---------------------:|
| **Train Accuracy** | 96.68% [96.60%, 96.77%] | 94.21% [94.02%, 94.40%] |
| **Test Accuracy** | 95.31% [95.16%, 95.46%] | 93.71% [93.42%, 94.00%] |
| **FC1 Spectral Norm** | 4.37 [4.24, 4.50] | 3.66 [3.53, 3.79] |
| **FC2 Spectral Norm** | 2.58 [2.42, 2.75] | 1.39 [1.30, 1.47] |
| **CKA (fc1_pre→fc1_post)** | 0.850 [0.828, 0.873] | 0.690 [0.669, 0.712] |
| **ROME fc1 Delta-Norm** | 14.49 (avg) | 7.18 (avg) |
| **ROME fc2 Delta-Norm** | 19.08 (avg) | 4.40 (avg) |
| **Loss Gap (corrupted, non-circular)** | — | −0.051 [−0.057, −0.045] |
| **Rank-5 Gap (ablation)** | — | 6.1 pp worse |

All ROME comparisons significant at p < 0.0001 across 10 seeds × 10 classes. CIFAR-10 replicates ROME finding (ratio 1.84×, all classes p < 0.05). See [RESULTS.md](RESULTS.md) for full tables.

### Additional Validations

| Analysis | Result |
|----------|--------|
| **FDR scaling** | Fisher discriminant ratio: 0.862 (h=16) → 0.387 (h=1024), decreases monotonically (replaces σ) |
| **Noise rate sweep (ROME fc2)** | Ratio scales monotonically: 3.37× (10%) → 4.34× (20%) → 5.98× (40%) |
| **Baseline: spectral norm fc2 ratio** | 1.86× (ROME is 2.3× more sensitive) |
| **Baseline: linear probe AUC** | 0.514 (barely above random — hidden activations don't encode corruption) |
| **ROME random baseline** | Signal ratio = ∞ (random rank-1 edits recover 0% vs ROME 10–22%) |
| **Multi-layer ROME** | Sequential (fc2→fc1) improves recovery by ~50% over single-layer |
| **Gradient anti-alignment** | +0.9944 [0.9926, 0.9961] at convergence (noise-dominated; anti-alignment occurs mid-training) |
| **0→8 recovery (5 seeds)** | +0.097 (updated from single-seed +0.014) |
| **CIFAR-10 Phase 3** | Loss gap −2.363 [−2.436, −2.291] (corrupted harder), GradAlign +0.441 |

## Central Findings

1. **Distortion localizes at the deepest pre-output interface** — CKA similarity drops significantly at the ReLU nonlinearity on MNIST (Δ=+0.160, p<0.001), shifting to the output-adjacent layer on CIFAR-10 (Δ=+0.103, p=0.009). The depth scales with architecture.
2. **FDR reveals true separability scaling** — Unlike σ (which conflates spread with dimensionality), FDR = tr(S_B)/tr(S_W) decreases with width, showing wider networks distribute class information across more dimensions.
3. **ROME delta-norm is a robust cross-architecture probe** — 10 seeds × 10 classes on MNIST and CIFAR-10 all show clean > corrupted at p<0.05. Ratio scales monotonically with noise rate (3.37× to 5.98×) and outperforms spectral norms (1.86×) and linear probes (AUC 0.514). Random baseline confirms structured edits are meaningful (signal ratio = ∞).
4. **Multi-layer ROME improves recovery** — Sequential editing of fc2 then fc1 recovers 22–40% vs 10–22% for single-layer, confirming distributed memorization.
5. **Wider networks distribute memorization** — Monosemanticity decreases with width; sparsity converges to 1.0 beyond h=128 (superposition hypothesis alignment).

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── model.py                  # MNISTNet + CIFAR10MLP
│   │   └── cifarnet.py              # CIFARNet (3072→256→128→10)
│   ├── utils/
│   │   ├── metrics.py               # CKA, FDR, monosemanticity, ROME utilities
│   │   └── stats.py                 # CI, paired t-test, multi-seed runner
│   ├── training/
│   │   ├── train_clean.py           # MNIST clean
│   │   ├── train_corrupted.py       # MNIST corrupted (saves corrupt_indices.npy)
│   │   ├── train_targeted_corrupted.py
│   │   ├── train_cifar.py           # CIFARNet combined clean/corrupted trainer
│   │   ├── train_cifar10_clean.py
│   │   ├── train_cifar10_corrupted.py
│   │   └── train_cifar10_scaling.py
│   ├── analysis/
│   │   ├── phase1_basic.py          # Weight norms, gradients, FDR
│   │   ├── phase2_representation.py # CKA, PCA, activation statistics
│   │   ├── phase3_influence.py      # Memorization metrics, gradient alignment
│   │   ├── phase4_rome.py           # ROME (single-layer)
│   │   ├── multiclass_rome.py       # Multi-class ROME + random baseline
│   │   ├── multilayer_rome.py       # Sequential + joint multi-layer ROME
│   │   ├── rank_ablation.py         # SVD rank ablation
│   │   ├── cifar_replication.py     # CIFAR-10 CKA + ROME + rank ablation rep.
│   │   ├── analyze_cifar10.py
│   │   ├── rome_cifar10.py
│   │   ├── analyze_cifar10_scaling.py
│   │   └── visualizations.py        # All publication figures
│   └── scaling/
│       ├── train_scaling.py
│       └── analyze_scaling.py        # FDR, monosemanticity, sparsity vs width
├── docs/
│   └── theoretical_propositions.md   # 3 formal propositions
├── paper/
│   ├── main.tex                      # LaTeX paper
│   └── related_work.bib
├── scripts/
│   ├── verify_consistency.py         # Cross-document number verification
│   └── verify_statistics.py          # CI and seed count verification
├── configs/experiment_config.yaml
├── tests/test_metrics.py            # 24 unit tests
├── reproduce_all.py                 # Single-command pipeline
├── RESULTS.md                       # Verified result tables
├── METHODOLOGY.md                   # 4-phase methodology
├── PAPER.md                         # Paper-to-code mapping
├── README.md
├── requirements.txt
└── environment.yml
```

## Quick Start

```bash
# Set up environment
conda env create -f environment.yml
conda activate memorization-analysis

# Run full MNIST pipeline (using existing checkpoints)
python reproduce_all.py --skip-training

# Run with CIFAR-10 validation
python reproduce_all.py --skip-training --skip-cifar10

# From scratch (training + analysis + figures)
python reproduce_all.py

# Run tests
python -m pytest tests/ -v          # 24/24 pass
```

## Pipeline

1. **Phase 1: Weight Geometry** — Spectral norms, Frobenius norms, gradient norms, FDR
2. **Phase 2: Representation Analysis** — CKA similarity between layers, activation statistics
3. **Phase 3: Influence Functions** — Non-circular memorization scoring, gradient alignment
4. **Phase 4: ROME Analysis** — Rank-One Model Editing, multi-class validation, random baseline, multi-layer ROME

## Configuration

All hyperparameters in `configs/experiment_config.yaml` — model dimensions, training epochs, learning rates, seed lists, and CIFAR-10 config block.

## GitHub Topics

When publishing this repo on GitHub, add these topics in the repo settings (Settings → Topics):

`neural-networks` `memorization` `interpretability` `cka` `rome` `mnist` `representation-similarity` `model-editing`

## Citation

```bibtex
@software{memorization_analysis_2026,
  title = {Structural Fingerprints of Label Memorization in Shallow Neural Networks},
  year = {2026},
  url = {https://github.com/shamiquekhan/four-phase-memorization-analysis}
}
```
