# Methodology: Memorization in Neural Networks

## Overview

Four-phase empirical analysis of how fully connected neural networks memorize training data. We compare models trained on clean MNIST against models trained with 20% label noise, measuring differences in weight geometry, representation similarity, memorization influence, and causal intervention recoverability.

## Model Architecture

All experiments use MNISTNet — a two-layer fully connected network:

```
Input (784) → Linear(784→16) → ReLU → Linear(16→10) → Output (10)
```

- Hidden dimension: 16 (default), scaling experiments test 32–1024
- Parameter count: 12,874
- Training: Adam, lr=0.001, batch=128, 20 epochs, cross-entropy loss

The small hidden dimension (16) forces the model into a compressed representation regime where individual neuron interpretability is feasible.

## Training Regimes

Two distinct corruption regimes are used, each serving a different analytical purpose:

### Regime A: Random Label Noise (Phases 1–3, Scaling)
- 20% of training labels are randomly flipped to a different class before training
- ~12,000 of 60,000 training samples corrupted per seed
- Studies *general memorization*: the model must memorize a random subset of incorrect labels while learning genuine features from the remaining 80%
- Used in Phases 1–3 (weight geometry, CKA, influence) and the scaling analysis
- Ground-truth corruption indices saved as `corrupt_indices.npy` for non-circular Phase 3 analysis

### Regime B: Targeted Class Swap (Phase 4 / Multi-class ROME)
- All samples of a specific source class are relabeled as a specific target class
- ~5,900–6,000 samples corrupted per class pair
- Four configurations tested: 7→1, 1→7, 5→6, 0→8
- Creates a controlled *backdoor* that ROME interventions can target and attempt to repair
- Used exclusively in Phase 4 (multi-class ROME validation)

### 3. Clean Training (Baseline)
Standard supervised learning on MNIST (60k train, 10k test). 10 random seeds for statistical rigor. All metrics are compared against this baseline to isolate memorization effects from normal learning dynamics.

## Four-Phase Analysis Pipeline

### Phase 1: Weight Geometry & Basic Metrics
- **Purpose**: Measure low-level differences in weight matrices and optimization dynamics
- **Metrics**:
  - Frobenius norm of weight matrices (`||W||_F`)
  - Spectral norm of weight matrices (`||W||_2`)
  - Gradient norm at the end of training
  - Training/Test accuracy and loss
- **Method**: Load final checkpoint, compute norms via PyTorch, aggregate with 95% CI

### Phase 2: Representation Similarity (CKA)
- **Purpose**: Quantify how representations differ between clean and memorized models
- **Metrics**:
  - Linear CKA similarity between adjacent layer pairs (input→fc1_pre, fc1_pre→fc1_post, fc1_post→output)
  - Activation sparsity (fraction of zero activations at ReLU output)
  - PCA explained variance of hidden representations
- **Method**: Extract activations on 5000 test samples, compute CKA via HSIC. Zero-centering applied before CKA computation.

### Phase 3: Influence & Memorization Metrics
- **Purpose**: Identify which training samples are memorized vs generalized, and which neurons specialize in memorization
- **Metrics**:
  - Per-sample loss distribution (memorized samples have higher loss under clean evaluation)
  - Memorized fraction (samples in top quartile of loss)
  - Loss gap (mean loss of memorized vs non-memorized samples)
- **Method**: Compute loss on training set, classify samples by loss quantile, use conjugate gradient for approximate influence estimation

### Phase 4: ROME (Rank-One Model Editing)
- **Purpose**: Apply and measure causal interventions to localize memorized associations
- **Approach**:
  - Compute rank-1 SVD updates to weight matrices that change the model's output for a specific class
  - Measure delta norm (magnitude of edit), effect on target class, and side effects on other classes
  - Apply ROME to both fc1 and fc2 layers independently
- **Multi-class ROME**: Apply targeted edits to the four corruption configs and measure recovery of source-class accuracy, side effects on other classes, and edit magnitude
- **Rank Ablation**: Replace weight matrices with rank-k SVD approximations (k=1..10) and measure accuracy degradation at each rank

## Scaling Analysis

- Hidden dimensions tested: [16, 32, 64, 128, 256, 512, 1024]
- Metrics tracked vs hidden dimension:
  - Davies-Bouldin index (class separability; dimension-invariant, replaces σ)
  - Monosemanticity fraction (fraction of neurons with max correlation > threshold)
  - Circuit size (neurons needed per class decision, via critical neuron ablation)
  - Network sparsity (fraction of near-zero weights)
  - Test accuracy
- Models trained for 100 epochs at each dimension with 10 random seeds

## Statistical Methodology

- **10 random seeds** for all experiments (seeds: 42, 123, 456, 789, 1024, 2048, 3141, 5555, 7777, 9999)
- **95% confidence intervals** computed via Student's t-distribution: `mean ± t_{0.975, n-1} * σ / sqrt(n)`
- All metrics reported as `mean [CI_low, CI_high]` across seeds

## Reproducibility

Full pipeline is orchestrated by `reproduce_all.py`:
```bash
# From scratch (training + analysis + figures)
python reproduce_all.py

# Using existing checkpoints (analysis + figures only)
python reproduce_all.py --skip-training
```

Configuration: `configs/experiment_config.yaml` (single source of truth for all hyperparameters).
