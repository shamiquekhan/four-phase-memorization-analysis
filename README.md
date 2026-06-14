# Structural Fingerprints of Label Memorization in Shallow Neural Networks

A systematic 4-phase analysis of how label memorization leaves structural fingerprints in shallow ReLU networks — spanning CKA representation drift, spectral geometry, circuit sparsity, influence functions, and rank-one model editing (ROME). Primary experiments on MNIST (784→16→10), validated on CIFAR-10 (3-layer MLP, 3072→512→256→10), with width scaling from 16 to 1024 hidden units.

## Key Results (MNIST, 10 seeds, 95% CI)

| Metric | Clean | Corrupted (20% noise) |
|--------|:-----:|:---------------------:|
| **Test Accuracy** | 95.31% [95.16%, 95.46%] | 93.71% [93.42%, 94.00%] |
| **FC1 Spectral Norm** | 4.37 [4.24, 4.50] | 3.66 [3.53, 3.79] |
| **FC2 Spectral Norm** | 2.58 [2.42, 2.75] | 1.39 [1.30, 1.47] |
| **CKA (fc1_pre→fc1_post)** | 0.850 [0.828, 0.873] | 0.690 [0.669, 0.712] |
| **ROME fc1 Delta-Norm** | 14.49 (avg) | 7.18 (avg) |
| **ROME fc2 Delta-Norm** | 19.08 (avg) | 4.40 (avg) |
| **Loss Gap (corrupted, non-circular)** | — | −0.051 [−0.057, −0.045] |
| **Rank-5 Gap (ablation)** | — | 6.1 pp worse |

All ROME comparisons significant at p < 0.0001 across 10 seeds × 10 classes. CIFAR-10 replicates ROME finding (ratio 1.84×, all classes p < 0.05). See [RESULTS.md](RESULTS.md) for full tables.

## Central Findings

1. **Distortion localizes at the deepest pre-output interface** — CKA similarity drops significantly at the ReLU nonlinearity on MNIST (Δ=+0.160, p<0.001), shifting to the output-adjacent layer on CIFAR-10 (Δ=+0.103, p=0.009). The depth scales with architecture.
2. **Spectral geometry is depth-dependent** — 2-layer MNIST shows suppression across all layers; 3-layer CIFAR-10 shows unchanged input layer but intensifying deeper layers.
3. **ROME delta-norm is a robust cross-architecture probe** — 10 seeds × 10 classes on MNIST and CIFAR-10 all show clean > corrupted at p<0.05.
4. **Wider networks distribute memorization** — Monosemanticity decreases with width; sparsity converges to 1.0 beyond h=128 (superposition hypothesis alignment).

## Project Structure

```
├── src/
│   ├── models/model.py              # MNISTNet + CIFAR10MLP
│   ├── utils/metrics.py             # CKA, DB index, monosemanticity
│   ├── utils/stats.py               # CI, paired t-test
│   ├── training/
│   │   ├── train_clean.py           # MNIST clean
│   │   ├── train_corrupted.py       # MNIST corrupted
│   │   ├── train_cifar10_clean.py   # CIFAR-10 clean
│   │   ├── train_cifar10_corrupted.py
│   │   └── train_cifar10_scaling.py # CIFAR-10 width scaling
│   ├── analysis/
│   │   ├── phase1_basic.py          # Weight norms, gradients
│   │   ├── phase2_representation.py # CKA, PCA, activations
│   │   ├── phase3_influence.py      # Memorization metrics
│   │   ├── phase4_rome.py           # ROME (single-layer)
│   │   ├── multiclass_rome.py       # Multi-class ROME validation
│   │   ├── rank_ablation.py         # SVD rank ablation
│   │   ├── analyze_cifar10.py       # CIFAR-10 Phase 1+2
│   │   ├── rome_cifar10.py          # CIFAR-10 ROME
│   │   ├── analyze_cifar10_scaling.py
│   │   └── visualizations.py        # Publication figures
│   └── scaling/
│       ├── train_scaling.py
│       └── analyze_scaling.py
├── docs/
│   └── theoretical_propositions.md  # 3 formal propositions
├── paper/
│   ├── main.tex                     # LaTeX paper (ICLR target)
│   └── related_work.bib             # 15-entry bibliography
├── configs/experiment_config.yaml   # All hyperparameters
├── tests/test_metrics.py            # 20 unit tests
├── outputs/                         # Results and checkpoints
├── reproduce_all.py                 # Single-command pipeline
├── RESULTS.md                       # Verified result tables
├── README.md
├── requirements.txt
└── environment.yml
```

## Quick Start

```bash
# Set up environment
conda env create -f environment.yml
conda activate memorization-analysis

# Run full MNIST pipeline
python reproduce_all.py

# Run with CIFAR-10 validation (requires CIFAR-10 checkpoints)
python reproduce_all.py --include-cifar10

# Run tests
python -m pytest tests/ -v          # 20/20 pass
```

## Pipeline

1. **Phase 1: Weight Geometry** — Spectral norms, Frobenius norms, gradient norms
2. **Phase 2: Representation Analysis** — CKA similarity between layers, activation statistics
3. **Phase 3: Influence Functions** — Non-circular memorization scoring (ground-truth corruption indices)
4. **Phase 4: ROME Analysis** — Rank-One Model Editing (Meng et al., 2022), multi-class validation, rank ablation

## Configuration

All hyperparameters in `configs/experiment_config.yaml` — model dimensions, training epochs, learning rates, seed lists, and CIFAR-10 config block.

## Citation

```bibtex
@software{memorization_analysis_2026,
  title = {Structural Fingerprints of Label Memorization in Shallow Neural Networks},
  year = {2026},
  url = {https://github.com/shamiquekhan/four-phase-memorization-analysis}
}
```
