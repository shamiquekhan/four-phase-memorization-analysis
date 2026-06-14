# Structural Fingerprints of Label Memorization in Shallow Neural Networks

A systematic 4-phase analysis of how label memorization leaves structural fingerprints in shallow ReLU networks — spanning CKA representation drift, σ geometry, circuit sparsity, influence functions, and rank-one model editing (ROME) across clean and corrupted MNIST, with scaling from 16 to 1024 hidden units.

## Key Results (10 seeds, 95% CI)

| Metric | Clean MNIST | Corrupted (20% noise) |
|--------|------------|----------------------|
| **Test Accuracy** | 95.31% [94.88, 95.74] | 93.81% [92.58, 95.04] |
| **FC1 Spectral Norm** | 12.56 [12.38, 12.74] | 10.28 [8.85, 11.71] |
| **FC2 Spectral Norm** | 5.00 [4.83, 5.17] | 2.76 [2.21, 3.31] |
| **CKA (input→fc1_pre)** | 0.712 [0.697, 0.727] | 0.693 [0.676, 0.710] |
| **CKA (fc1_pre→fc1_post)** | 0.850 [0.828, 0.872] | 0.690 [0.669, 0.711] |
| **Loss Gap (corrupted)** | — | -0.051 [-0.057, -0.045] |
| **ROME Recovery (multi-class)** | — | +0.097 to +0.217 |
| **Rank for 90% of Full Acc** | 8/10 | 8/10 |


## Project Structure

```
├── src/
│   ├── models/model.py        # MNISTNet definition
│   ├── utils/stats.py         # CI computation utilities
│   ├── training/              # Training scripts
│   │   ├── train_clean.py
│   │   └── train_corrupted.py
│   ├── analysis/              # 4-phase analysis
│   │   ├── phase1_basic.py          # Weight norms, gradients
│   │   ├── phase2_representation.py # CKA, PCA, activations
│   │   ├── phase3_influence.py      # Memorization metrics
│   │   ├── phase4_rome.py           # ROME analysis
│   │   ├── multiclass_rome.py       # Multi-class ROME validation
│   │   ├── rank_ablation.py         # Rank ablation study
│   │   └── visualizations.py        # Publication figures
│   └── scaling/               # Scaling experiments
│       ├── train_scaling.py
│       └── analyze_scaling.py
├── configs/experiment_config.yaml   # All hyperparameters
├── tests/test_metrics.py            # Unit tests
├── outputs/                         # Results and checkpoints
├── reproduce_all.py                 # Single-command pipeline
├── requirements.txt
└── environment.yml
```

## Quick Start

```bash
# Set up environment
conda env create -f environment.yml
conda activate memorization-analysis

# Run full pipeline (training + analysis + figures)
python reproduce_all.py

# Or run individual steps:
python src/training/train_clean.py --seed 42
python src/training/train_corrupted.py --seed 42 --noise-rate 0.2
python src/analysis/phase1_basic.py --checkpoint-dir outputs/clean
python src/analysis/visualizations.py

# Run tests
python -m pytest tests/ -v
```

## Pipeline

1. **Phase 1: Basic Analysis** — Epoch-wise accuracy, loss curves, weight norms (Frobenius, spectral), gradient norms
2. **Phase 2: Representation Analysis** — CKA similarity between layers, activation statistics (sparsity, mean, std), PCA
3. **Phase 3: Influence Functions** — Per-sample memorization scores, influence estimation via conjugate gradient
4. **Phase 4: ROME Analysis** — Rank-One Model Editing to localize memorized associations, multi-class validation

## Configuration

All hyperparameters are in `configs/experiment_config.yaml`:
- Model: input/hidden/output dimensions, activation function
- Training: epochs, learning rate, batch size
- Analysis: confidence intervals, seeds

## Reproducing Key Results

```bash
# Full 10-seed pipeline
python reproduce_all.py --seeds 10

# Skip training to just run analysis (requires existing checkpoints)
python reproduce_all.py --skip-training

# Generate publication figures
python src/analysis/visualizations.py
```

## Citation

If you use this code, please cite:
```
@software{memorization_analysis_2026,
  title = {Structural Fingerprints of Label Memorization in Shallow Neural Networks},
  year = {2026},
  url = {https://github.com/anomalyco/ml-research}
}
```
