"""Visualization utilities for analysis figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import torch
import numpy as np
from pathlib import Path
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
COLORS = sns.color_palette("husl", 10)


# Figure 1: Four-Phase Framework Overview
def figure1_framework(output_path):
    """Figure 1: Four-Phase Framework diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    phases = [
        ('Phase 1: Geometric', 'Class separability (σ)\nWithin/between class\ndistance ratio', '#E74C3C'),
        ('Phase 2: Representational', 'Monosemanticity\nNeuron-class correlations\nCKA similarity', '#3498DB'),
        ('Phase 3: Algorithmic', 'Circuit sparsity\nCritical neuron ablation\nInfluence functions', '#2ECC71'),
        ('Phase 4: Causal', 'ROME rank-1 editing\nCausal intervention\nRecovery verification', '#9B59B6'),
    ]

    for i, (title, desc, color) in enumerate(phases):
        x = 1.5 + i * 2.2
        rect = mpatches.FancyBboxPatch((x, 2.5), 1.8, 2.5, boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.9, 4.5, title, ha='center', va='center', fontsize=11, fontweight='bold', color=color)
        ax.text(x + 0.9, 3.3, desc, ha='center', va='center', fontsize=8, color='#333')
        if i < 3:
            ax.annotate('', xy=(x + 1.8, 3.75), xytext=(x + 2.0, 3.75),
                       arrowprops=dict(arrowstyle='->', color='#888', lw=2))

    ax.text(5, 1.0, 'Input (784D) → Hidden (128D) → Output (10D)', ha='center', fontsize=12,
           fontweight='bold', bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#ccc'))
    ax.text(5, 0.3, 'Training: Clean MNIST / Corrupted MNIST (20% label noise) • Metrics: Mean ± 95% CI across 10 seeds',
           ha='center', fontsize=9, color='#666')

    fig.suptitle('Figure 1: Four-Phase Memorization Analysis Framework', fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(output_path / 'figure1_framework.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# Figure 2: ROME Pipeline Diagram
def figure2_rome_pipeline(output_path):
    """Figure 2: ROME pipeline diagram."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    steps = [
        ('Step 1: Train', 'MNIST\non clean\n& corrupted\ndata', '#3498DB'),
        ('Step 2: Identify', 'Find broken\nclass with\nlow accuracy', '#E74C3C'),
        ('Step 3: Edit', 'Compute rank-1\nupdate on fc2\nweight matrix', '#2ECC71'),
        ('Step 4: Validate', 'Measure\nrecovery &\nside effects', '#9B59B6'),
    ]

    for ax, (title, desc, color) in zip(axes, steps):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        rect = mpatches.FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.12, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, 0.75, title, ha='center', va='center', fontsize=10, fontweight='bold', color=color)
        ax.text(0.5, 0.4, desc, ha='center', va='center', fontsize=9, color='#333')
        if ax != axes[-1]:
            ax.annotate('', xy=(0.95, 0.5), xytext=(1.02, 0.5),
                       arrowprops=dict(arrowstyle='->', color='#888', lw=2), transform=ax.transAxes)

    fig.suptitle('Figure 2: ROME Causal Intervention Pipeline', fontsize=14, fontweight='bold', y=1.05)
    fig.tight_layout()
    fig.savefig(output_path / 'figure2_rome_pipeline.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# Figure 3: t-SNE/PCA Manifold Progression
def figure3_tsne_manifold(model, test_loader, device, output_path, n_samples=2000):
    """Figure 3: t-SNE manifold progression across input/hidden/output layers."""
    X_input, X_hidden, X_output, labels = [], [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)
            h = model.get_hidden(x)
            out = model(x)
            X_input.append(x_flat.cpu().numpy())
            X_hidden.append(h.cpu().numpy())
            X_output.append(out.cpu().numpy())
            labels.append(y.numpy())
            if sum(len(l) for l in labels) >= n_samples:
                break

    X_in = np.concatenate(X_input)[:n_samples]
    X_hid = np.concatenate(X_hidden)[:n_samples]
    X_out = np.concatenate(X_output)[:n_samples]
    y = np.concatenate(labels)[:n_samples]

    sigmas = []
    for X in [X_in, X_hid, X_out]:
        within, between = [], []
        for c in range(10):
            ca = X[y == c]
            oa = X[y != c]
            if len(ca) < 2:
                continue
            n = min(len(ca), 200)
            idx = np.random.choice(len(ca), n, replace=False)
            s = ca[idx]
            d1 = np.sqrt(((s[:, None] - s[None, :])**2).sum(-1))
            within.append(d1[np.triu_indices(n, k=1)].mean())
            m = min(len(oa), 200)
            os = oa[np.random.choice(len(oa), m)]
            d2 = np.sqrt(((s[:, None] - os[None, :])**2).sum(-1))
            between.append(d2.mean())
        sigmas.append(np.mean(within) / max(np.mean(between), 1e-8))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, X, title, sigma in zip(
        axes, [X_in, X_hid, X_out],
        ['Input Space (784D)', 'Hidden Space (128D)', 'Output Space (10D)'],
        sigmas
    ):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(X)
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                            c=y, cmap='tab10', alpha=0.6, s=5)
        ax.set_title(f'{title}\nσ = {sigma:.3f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')

    fig.colorbar(scatter, ax=axes[-1], label='Digit Class')
    fig.suptitle('Figure 3: t-SNE Manifold Progression Across Layers\n'
                 '(Geometric disentanglement: σ decreases from input to output)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / 'figure3_tsne_manifold.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# Figure 4: Neuron Weight Receptive Fields
def figure4_neuron_weights(model, output_path, top_n=16):
    """Figure 4: Neuron weight visualization as 28x28 receptive fields."""
    W1 = model.fc1.weight.data.cpu().numpy()
    n_neurons = min(W1.shape[0], top_n)
    cols = 8
    rows = (n_neurons + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 2 * rows))
    for idx in range(rows * cols):
        ax = axes.flatten()[idx]
        if idx >= n_neurons:
            ax.axis('off')
            continue
        weights = W1[idx].reshape(28, 28)
        vmax = abs(weights).max()
        ax.imshow(weights, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'N{idx+1}', fontsize=8)
        ax.axis('off')

    fig.suptitle('Figure 4: Hidden Neuron Receptive Fields (fc1 Weight Vectors)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path / 'figure4_neuron_weights.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# Figure 5: Neuron-Class Correlation Heatmap
def figure5_correlation_heatmap(model, test_loader, device, output_path):
    """Figure 5: Neuron-class Pearson correlation heatmap."""
    hidden_acts, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            h = model.get_hidden(x).cpu().numpy()
            hidden_acts.append(h)
            all_labels.append(y.numpy())

    hidden_acts = np.concatenate(hidden_acts)
    all_labels = np.concatenate(all_labels)
    n_neurons = hidden_acts.shape[1]
    n_classes = 10
    correlations = np.zeros((n_neurons, n_classes))

    for n in range(n_neurons):
        neuron_acts = hidden_acts[:, n]
        for c in range(n_classes):
            r, _ = pearsonr(neuron_acts, (all_labels == c).astype(float))
            correlations[n, c] = abs(r)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlations, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[f'Class {i}' for i in range(n_classes)],
                yticklabels=[f'Neuron {i+1}' for i in range(n_neurons)],
                ax=ax, vmin=0, vmax=1)
    ax.set_title('Figure 5: Neuron-Class Pearson Correlation Matrix\n'
                 '(Rows = Neurons, Columns = Classes)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path / 'figure5_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# Figure 6: Circuit Map
def figure6_circuit_map(model, test_loader, device, output_path, threshold=0.02):
    """Figure 6: Circuit map showing critical neuron-class connections."""
    hidden_size = model.hidden_dim if hasattr(model, 'hidden_dim') else 128
    n_classes = 10
    max_neurons = min(hidden_size, 16)

    relu_module = next((m for m in model.modules() if isinstance(m, torch.nn.ReLU)), None)
    if relu_module is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Circuit map requires ReLU module', ha='center', fontsize=14)
        fig.savefig(output_path / 'figure6_circuit_map.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        return

    baseline, counts = {}, {}
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            for c in range(n_classes):
                m = y == c
                baseline[c] = baseline.get(c, 0) + (preds[m] == y[m]).sum().item()
                counts[c] = counts.get(c, 0) + m.sum().item()
    for c in range(n_classes):
        baseline[c] /= max(counts[c], 1)

    critical = np.zeros((max_neurons, n_classes), dtype=int)
    for n in range(max_neurons):
        def make_hook(ni):
            return lambda mod, inp, out: out.clone().scatter_(1, torch.tensor([[ni]]).to(out.device), 0)
        hook = relu_module.register_forward_hook(make_hook(n))
        ablated = {}
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                for c in range(n_classes):
                    m = y == c
                    ablated[c] = ablated.get(c, 0) + (preds[m] == y[m]).sum().item()
        hook.remove()
        for c in range(n_classes):
            ablated[c] /= max(counts[c], 1)
            if baseline[c] - ablated[c] > threshold:
                critical[n, c] = 1

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(critical, cmap=plt.cm.Blues, aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Neuron Index', fontsize=12)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(max_neurons))
    ax.set_xticklabels([str(i) for i in range(n_classes)])
    ax.set_yticklabels([f'N{i+1}' for i in range(max_neurons)])
    ax.set_title('Figure 6: Circuit Map — Critical Neuron-Class Connections\n'
                 '(Blue = Critical, White = Non-critical, threshold=2% accuracy drop)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path / 'figure6_circuit_map.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# Figure 7: ROME Accuracy Recovery Bar Chart
def figure7_rome_barchart(rome_results, output_path):
    """Figure 7: ROME accuracy recovery before/after by class."""
    if not rome_results:
        return

    labels = list(rome_results.keys())
    n_configs = len(labels)
    before_means, after_means = [], []

    for label in labels:
        data = rome_results[label]
        recovery_mean = float(np.mean(data.get('recovery', [0])))
        pre_list = data.get('pre_accs', []) if isinstance(data.get('pre_accs'), list) else []

        if pre_list and isinstance(pre_list[0], dict):
            pre_accs = pre_list[0]
            source_class = int(label.split('→')[0])
            before = pre_accs.get(str(source_class), 0)
        else:
            before = 0.0

        before_means.append(max(0, before * 100))
        after_means.append(max(0, min(100, (before + recovery_mean) * 100)))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_configs)
    width = 0.35

    ax.bar(x - width/2, before_means, width, label='Before ROME', color='#E74C3C', alpha=0.8)
    ax.bar(x + width/2, after_means, width, label='After ROME', color='#2ECC71', alpha=0.8)

    ax.set_xlabel('Corruption Configuration (Source→Target)')
    ax.set_ylabel('Class Accuracy (%)')
    ax.set_title('Figure 7: ROME Recovery — Per-Class Accuracy Before vs After Editing', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    for i in range(n_configs):
        ax.text(i - width/2, before_means[i] + 1, f'{before_means[i]:.0f}%', ha='center', fontsize=8)
        ax.text(i + width/2, after_means[i] + 1, f'{after_means[i]:.0f}%', ha='center', fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path / 'figure7_rome_recovery.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# Figure 8: Scaling Analysis
def figure8_scaling_analysis(scaling_results, output_path):
    """Figure 8: 4-panel scaling analysis showing σ, monosemanticity, circuit size, sparsity vs width."""
    if not scaling_results:
        return

    hidden_dims = sorted([int(k) for k in scaling_results.keys()])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_cfg = [
        ('sigma', 'σ (Within/Between Ratio)', 'lower is better'),
        ('mono_fraction', 'Monosemantic Fraction', 'higher is better'),
        ('circuit_size', 'Mean Circuit Size (neurons/class)', 'smaller = sparser'),
        ('sparsity', 'Network Sparsity', 'higher = sparser'),
    ]

    for (metric, ylabel, note), ax in zip(metrics_cfg, axes.flat):
        means = [np.mean(scaling_results[str(h)].get(metric, [0])) for h in hidden_dims]
        ax.plot(hidden_dims, means, 'o-', linewidth=2)
        ax.set_xlabel('Hidden Dimension')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel}\n({note})', fontsize=11)
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 8: Scaling Analysis — Interpretability Metrics vs Model Width', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / 'figure8_scaling_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# Supplementary Figures
def figure_training_curves(clean_history, corrupted_history, seeds, output_path, config):
    """Supplementary: Training curves comparing clean vs corrupted."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = range(1, config['training']['epochs'] + 1)

    for idx, (histories, label, color) in enumerate([
        (clean_history, 'Clean', 'C0'), (corrupted_history, 'Corrupted', 'C1')
    ]):
        ax = axes[0, 0]
        losses = [histories[s]['train_loss'] for s in seeds if s in histories]
        if losses:
            m = np.mean(losses, axis=0)
            s = np.std(losses, axis=0)
            ax.plot(epochs, m, color=color, label=label)
            ax.fill_between(epochs, m - s, m + s, alpha=0.2, color=color)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Train Loss'); ax.set_title('Training Loss'); ax.legend()

        ax = axes[0, 1]
        accs = [histories[s]['test_acc'] for s in seeds if s in histories]
        if accs:
            m = np.mean(accs, axis=0)
            s = np.std(accs, axis=0)
            ax.plot(epochs, m, color=color, label=label)
            ax.fill_between(epochs, m - s, m + s, alpha=0.2, color=color)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Test Accuracy (%)'); ax.set_title('Test Accuracy'); ax.legend()

    ax = axes[1, 0]
    clean_final = [histories[s]['test_acc'][-1] for s in seeds if s in clean_history]
    corr_final = [histories[s]['test_acc'][-1] for s in seeds if s in corrupted_history]
    ax.boxplot([clean_final, corr_final], tick_labels=['Clean', 'Corrupted'])
    ax.set_ylabel('Final Test Accuracy (%)'); ax.set_title('Across-Seed Variability')

    ax = axes[1, 1]; ax.axis('off')
    ax.text(0.1, 0.5, f"Clean: {np.mean(clean_final):.1f}% ± {np.std(clean_final):.1f}%\n"
                     f"Corrupted: {np.mean(corr_final):.1f}% ± {np.std(corr_final):.1f}%",
           fontsize=14, va='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    fig.suptitle('Supplementary: Training Dynamics — Clean vs Corrupted MNIST', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / 'supp_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def figure_weight_analysis(clean_results, corrupted_results, output_path):
    """Supplementary: Weight norm comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pairs = [
        ('fc1_frobenius', 'FC1 Frobenius Norm', axes[0, 0]),
        ('fc2_frobenius', 'FC2 Frobenius Norm', axes[0, 1]),
        ('fc1_spectral', 'FC1 Spectral Norm', axes[1, 0]),
    ]
    for metric, ylabel, ax in pairs:
        clean_vals = [r[metric] for r in clean_results.values() if metric in r]
        corr_vals = [r[metric] for r in corrupted_results.values() if metric in r]
        if clean_vals and corr_vals:
            ax.boxplot([clean_vals, corr_vals], tick_labels=['Clean', 'Corrupted'])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    axes[1, 1].axis('off')
    fig.suptitle('Supplementary: Weight Norm Analysis', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / 'supp_weight_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def figure_cka_heatmap(cka_results_clean, cka_results_corrupted, output_path):
    """Supplementary: CKA similarity heatmaps."""
    layers = ['input', 'fc1_pre', 'fc1_post', 'output']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (results, title) in enumerate([(cka_results_clean, 'Clean MNIST'), (cka_results_corrupted, 'Corrupted MNIST')]):
        ax = axes[idx]
        mat = np.eye(len(layers))
        for i, l1 in enumerate(layers):
            for j, l2 in enumerate(layers):
                if i < j:
                    key = f'{l1}_{l2}'
                    vals = [results[s]['cka'].get(key, 0) for s in results]
                    mat[i, j] = np.mean(vals)
                    mat[j, i] = mat[i, j]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(layers)))
        ax.set_yticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45)
        ax.set_yticklabels(layers)
        ax.set_title(title)
        for i in range(len(layers)):
            for j in range(len(layers)):
                ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                       color='white' if mat[i, j] > 0.5 else 'black')
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.suptitle('Supplementary: CKA Representation Similarity', fontsize=14, y=1.05)
    fig.tight_layout()
    fig.savefig(output_path / 'supp_cka_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def figure_rome_heatmap(rome_results, output_path, title='ROME Edit Effects'):
    """Supplementary: ROME edit effects heatmap."""
    n_classes = 10
    mat = np.zeros((n_classes, n_classes))
    for tc in range(n_classes):
        for sk in rome_results:
            if tc in rome_results[sk]:
                res = rome_results[sk][tc]
                bl = res['baseline']
                ed = res['scales'].get(1.0, {})
                for c in range(n_classes):
                    mat[tc, c] += ed.get(c, bl.get(c, 0)) - bl.get(c, 0)
        mat[tc] /= max(len(rome_results), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = max(abs(mat).max(), 1e-8)
    ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xlabel('Evaluated Class'); ax.set_ylabel('Edited Class'); ax.set_title(title)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                   color='white' if abs(mat[i, j]) > 0.5 * vmax else 'black')
    fig.tight_layout()
    fig.savefig(output_path / 'supp_rome_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def figure_rank_ablation(rank_results, output_path):
    """Supplementary: Rank ablation analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for sk in rank_results:
        S = rank_results[sk]['singular_values']
        ax.plot(range(1, len(S) + 1), S, '-o', alpha=0.4, markersize=3)
    ax.set_xlabel('Singular Value Index'); ax.set_ylabel('Magnitude')
    ax.set_title('Singular Value Spectrum'); ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    c_ranks = []
    for sk in rank_results:
        r = rank_results[sk]['ranks']
        ta = rank_results[sk]['test_acc']
        ax.plot(r, ta, '-o', alpha=0.4, markersize=3)
        fa = ta[-1]
        for rr, aa in zip(r, ta):
            if aa < 0.9 * fa:
                c_ranks.append(rr)
                break
    if c_ranks:
        ax.axvline(np.mean(c_ranks), color='red', ls='--', label=f'Mean critical rank: {np.mean(c_ranks):.0f}')
    ax.set_xlabel('Rank'); ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Accuracy vs Rank'); ax.grid(True, alpha=0.3)
    fig.suptitle('Supplementary: Rank Ablation Analysis', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / 'supp_rank_ablation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# Main Orchestrator
def generate_all_figures(results_dir, output_path, config):
    results_dir = Path(results_dir)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    def _safe_load(path):
        p = Path(path)
        return json.load(open(p)) if p.exists() else None

    def _extract_phase1(d):
        if not d:
            return {}
        result = {}
        for sk in d:
            v = d[sk]
            if isinstance(v, list) and len(v) > 0:
                result[sk] = v[0]
        return result

    # Load available data
    phase1_clean = _extract_phase1(_safe_load(results_dir / 'analysis/phase1_clean/phase1_results.json'))
    phase1_corrupted = _extract_phase1(_safe_load(results_dir / 'analysis/phase1_corrupted/phase1_results.json'))
    phase2_clean = _safe_load(results_dir / 'analysis/phase2_clean/phase2_results.json') or {}
    phase2_corrupted = _safe_load(results_dir / 'analysis/phase2_corrupted/phase2_results.json') or {}
    rome_results = _safe_load(results_dir / 'analysis/phase4_clean/phase4_results.json') or {}
    rome_multi = _safe_load(results_dir / 'analysis/multiclass_rome/multiclass_rome_results.json') or {}

    print("Generating figures...")

    # Figure 1: Framework diagram (always generated)
    figure1_framework(output_path)

    # Figure 2: ROME pipeline diagram (always generated)
    figure2_rome_pipeline(output_path)

    # Figures 3-8: require trained model
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.model import MNISTNet
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=500)

        model_ckpts = list(Path(results_dir).glob('clean/seed_0/final_model.pt'))
        if model_ckpts:
            ckpt_data = torch.load(model_ckpts[0], map_location=device)
            arch = ckpt_data.get('architecture', {})
            h_dim = arch.get('hidden_dim', config.get('model', {}).get('hidden_dim', 16))
            model = MNISTNet(784, h_dim, 10).to(device)
            model.load_state_dict(ckpt_data['model_state_dict'])
            model.eval()

            figure3_tsne_manifold(model, test_loader, device, output_path)
            figure4_neuron_weights(model, output_path)
            figure5_correlation_heatmap(model, test_loader, device, output_path)
            figure6_circuit_map(model, test_loader, device, output_path)
        else:
            print("  Skipping figures 3-6: no trained model found")
    except Exception as e:
        print(f"  Skipping figures 3-6: {e}")

    # Figure 7: ROME recovery bar chart
    if rome_multi:
        figure7_rome_barchart(rome_multi, output_path)
    else:
        print("  Skipping figure 7: no ROME multi-class results")

    # Figure 8: Scaling analysis
    scaling_summary = _safe_load(results_dir / 'analysis/scaling/scaling_analysis.json')
    if scaling_summary:
        figure8_scaling_analysis(scaling_summary, output_path)
    else:
        print("  Skipping figure 8: no scaling analysis results")

    # Supplementary figures
    print("  Generating supplementary figures...")
    if phase1_clean and phase1_corrupted:
        figure_training_curves(phase1_clean, phase1_corrupted, list(range(10)), output_path, config)
        figure_weight_analysis(phase1_clean, phase1_corrupted, output_path)
    if phase2_clean and phase2_corrupted:
        figure_cka_heatmap(phase2_clean, phase2_corrupted, output_path)
    if rome_results:
        figure_rome_heatmap(rome_results, output_path)

    print(f"Figures saved to {output_path}")


if __name__ == '__main__':
    import argparse, yaml
    parser = argparse.ArgumentParser(description='Generate all figures')
    parser.add_argument('--results-dir', type=str, default='outputs')
    parser.add_argument('--output-dir', type=str, default='outputs/figures')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    generate_all_figures(args.results_dir, args.output_dir, config)
