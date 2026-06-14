"""
CIFAR-10 width scaling analysis: monosemanticity, σ (Davies-Bouldin), circuit sparsity.
Replicates Phase scaling findings on CIFAR-10 across widths 64, 128, 256, 512.
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import compute_monosemanticity, compute_circuit_sparsity, extract_hidden_activations
from utils.stats import compute_ci


def compute_davies_bouldin(activations, labels):
    """Davies-Bouldin index - dimension-invariant cluster separability metric.
    Lower is better (more separated)."""
    classes = np.unique(labels)
    centroids = []
    within_dists = []
    for c in classes:
        mask = labels == c
        class_acts = activations[mask]
        centroid = class_acts.mean(0)
        centroids.append(centroid)
        within_dists.append(np.mean(np.linalg.norm(class_acts - centroid, axis=1)))
    centroids = np.array(centroids)
    n_classes = len(classes)
    db_sum = 0
    for i in range(n_classes):
        max_ratio = 0
        for j in range(n_classes):
            if i == j:
                continue
            centroid_dist = np.linalg.norm(centroids[i] - centroids[j])
            ratio = (within_dists[i] + within_dists[j]) / (centroid_dist + 1e-8)
            max_ratio = max(max_ratio, ratio)
        db_sum += max_ratio
    return db_sum / n_classes


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint['architecture']
    from models.model import CIFAR10MLP
    model = CIFAR10MLP(
        input_dim=arch.get('input_dim', 3072),
        hidden1=arch.get('hidden1', 512),
        hidden2=arch.get('hidden2', 256),
        output_dim=arch.get('output_dim', 10),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def get_cifar10_loader(batch_size=256, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    dataset = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 scaling analysis')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/cifar10/scaling')
    parser.add_argument('--output-dir', type=str, default='outputs/cifar10/analysis/scaling')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 128, 256, 512])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_cifar10_loader()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for hdim in args.hidden_dims:
        hdim_results = {s: {} for s in args.seeds}
        for seed in args.seeds:
            ckpt_path = Path(args.checkpoint_dir) / f"hidden_{hdim}" / f"seed_{seed}" / 'final_model.pt'
            if not ckpt_path.exists():
                print(f"Checkpoint not found: {ckpt_path}")
                continue

            model = load_model(str(ckpt_path), device)

            hidden_acts, labels = extract_hidden_activations(model, loader, device, max_samples=5000)

            mono_frac, max_corrs = compute_monosemanticity(hidden_acts, labels, threshold=0.5)
            db_index = compute_davies_bouldin(hidden_acts, labels)

            hdim_results[seed] = {
                'monosemanticity': mono_frac,
                'davies_bouldin': db_index,
                'test_acc': checkpoint.get('metadata', {}).get('test_acc', None),
            }
            print(f"  h={hdim} seed={seed}: mono={mono_frac:.4f}, DB={db_index:.4f}")

        all_results[str(hdim)] = hdim_results

        hdim_vals = [hdim_results[s] for s in args.seeds if hdim_results[s]]
        if hdim_vals:
            mono_vals = [r['monosemanticity'] for r in hdim_vals]
            db_vals = [r['davies_bouldin'] for r in hdim_vals]
            m_mean, m_lo, m_hi = compute_ci(mono_vals)
            db_mean, db_lo, db_hi = compute_ci(db_vals)
            print(f"\n  h={hdim} aggregate: mono={m_mean:.4f} [{m_lo:.4f}, {m_hi:.4f}], "
                  f"DB={db_mean:.4f} [{db_lo:.4f}, {db_hi:.4f}]\n")

    with open(output_dir / 'cifar10_scaling_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
