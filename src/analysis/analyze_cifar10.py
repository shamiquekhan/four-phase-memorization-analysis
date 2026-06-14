"""
CIFAR-10 validation analysis: Phase 1 (spectral norms) and Phase 2 (CKA similarity).
Replicates key MNIST findings on CIFAR-10 with a 3-layer MLP.
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
from models.model import CIFAR10MLP
from utils.metrics import linear_cka
from utils.stats import compute_ci, paired_t_test


def get_cifar10_loader(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    dataset = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_activations(model, dataloader, device, max_samples=5000):
    model.eval()
    all_layers = {k: [] for k in ['input', 'fc1_pre_activation', 'fc1_post_activation', 'fc2_pre_activation', 'fc2_post_activation', 'output']}
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            out = model.forward_with_all_layers(data)
            for k in all_layers:
                all_layers[k].append(out[k].cpu())
            if sum(x.size(0) for x in all_layers['input']) >= max_samples:
                break
    return {k: torch.cat(v)[:max_samples] for k, v in all_layers.items()}


def compute_weight_norms(model):
    with torch.no_grad():
        return {
            'fc1_frobenius': model.fc1.weight.data.norm(p='fro').item(),
            'fc2_frobenius': model.fc2.weight.data.norm(p='fro').item(),
            'fc3_frobenius': model.fc3.weight.data.norm(p='fro').item(),
            'fc1_spectral': torch.linalg.norm(model.fc1.weight.data, ord=2).item(),
            'fc2_spectral': torch.linalg.norm(model.fc2.weight.data, ord=2).item(),
            'fc3_spectral': torch.linalg.norm(model.fc3.weight.data, ord=2).item(),
        }


def analyze_model(model, dataloader, device):
    acts = get_activations(model, dataloader, device)
    weight_norms = compute_weight_norms(model)

    cka_pairs = {
        'input_fc1_pre': linear_cka(acts['input'], acts['fc1_pre_activation']),
        'fc1_pre_fc1_post': linear_cka(acts['fc1_pre_activation'], acts['fc1_post_activation']),
        'fc1_post_fc2_pre': linear_cka(acts['fc1_post_activation'], acts['fc2_pre_activation']),
        'fc2_pre_fc2_post': linear_cka(acts['fc2_pre_activation'], acts['fc2_post_activation']),
        'fc2_post_output': linear_cka(acts['fc2_post_activation'], acts['output']),
    }

    activation_stats = {}
    for name, act in acts.items():
        activation_stats[name] = {
            'mean': act.mean().item(),
            'std': act.std().item(),
            'sparsity': (act == 0).float().mean().item(),
        }

    return {
        'cka': cka_pairs,
        'weight_norms': weight_norms,
        'activation_stats': activation_stats,
    }


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 validation analysis (Phase 1 + 2)')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--clean-dir', type=str, default='outputs/cifar10/clean')
    parser.add_argument('--corrupted-dir', type=str, default='outputs/cifar10/corrupted')
    parser.add_argument('--output-dir', type=str, default='outputs/cifar10/analysis')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_cifar10_loader()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_results = {}
    corrupted_results = {}

    for seed in args.seeds:
        ckpt_clean = Path(args.clean_dir) / f"seed_{seed}" / 'final_model.pt'
        if ckpt_clean.exists():
            model = CIFAR10MLP.load_checkpoint(str(ckpt_clean), device)
            clean_results[seed] = analyze_model(model, loader, device)
            print(f"seed={seed} clean: done")
        else:
            print(f"seed={seed} clean: checkpoint not found")

        ckpt_corr = Path(args.corrupted_dir) / f"noise_0.2" / f"seed_{seed}" / 'final_model.pt'
        if ckpt_corr.exists():
            model = CIFAR10MLP.load_checkpoint(str(ckpt_corr), device)
            corrupted_results[seed] = analyze_model(model, loader, device)
            print(f"seed={seed} corrupted: done")

    print("\n=== CIFAR-10 CKA Comparison: Clean vs Corrupted ===")
    cka_keys = ['input_fc1_pre', 'fc1_pre_fc1_post', 'fc1_post_fc2_pre', 'fc2_pre_fc2_post', 'fc2_post_output']
    for key in cka_keys:
        clean_vals = [clean_results[s]['cka'][key] for s in args.seeds if s in clean_results]
        corr_vals = [corrupted_results[s]['cka'][key] for s in args.seeds if s in corrupted_results]
        if clean_vals and corr_vals:
            c_mean, c_lo, c_hi = compute_ci(clean_vals)
            r_mean, r_lo, r_hi = compute_ci(corr_vals)
            delta = c_mean - r_mean
            t_stat, p_val = paired_t_test(clean_vals, corr_vals)
            print(f"{key:25s}  Clean: {c_mean:.4f} [{c_lo:.4f}, {c_hi:.4f}]  "
                  f"Corrupted: {r_mean:.4f} [{r_lo:.4f}, {r_hi:.4f}]  "
                  f"Δ={delta:+.4f}  p={p_val:.4f}")

    print("\n=== CIFAR-10 Weight Norms: Clean vs Corrupted ===")
    norm_keys = ['fc1_spectral', 'fc2_spectral', 'fc3_spectral']
    for key in norm_keys:
        clean_vals = [clean_results[s]['weight_norms'][key] for s in args.seeds if s in clean_results]
        corr_vals = [corrupted_results[s]['weight_norms'][key] for s in args.seeds if s in corrupted_results]
        if clean_vals and corr_vals:
            c_mean, c_lo, c_hi = compute_ci(clean_vals)
            r_mean, r_lo, r_hi = compute_ci(corr_vals)
            t_stat, p_val = paired_t_test(clean_vals, corr_vals)
            print(f"{key:25s}  Clean: {c_mean:.4f} [{c_lo:.4f}, {c_hi:.4f}]  "
                  f"Corrupted: {r_mean:.4f} [{r_lo:.4f}, {r_hi:.4f}]  p={p_val:.4f}")

    print("\n=== CIFAR-10 Activation Statistics ===")
    for layer in ['fc1_pre_activation', 'fc1_post_activation', 'fc2_pre_activation', 'fc2_post_activation']:
        for stat in ['mean', 'sparsity']:
            clean_vals = [clean_results[s]['activation_stats'][layer][stat] for s in args.seeds if s in clean_results]
            corr_vals = [corrupted_results[s]['activation_stats'][layer][stat] for s in args.seeds if s in corrupted_results]
            if clean_vals and corr_vals:
                c_mean, c_lo, c_hi = compute_ci(clean_vals)
                r_mean, r_lo, r_hi = compute_ci(corr_vals)
                t_stat, p_val = paired_t_test(clean_vals, corr_vals)
                print(f"{layer:15s} {stat:10s}  Clean: {c_mean:.4f} [{c_lo:.4f}, {c_hi:.4f}]  "
                      f"Corrupted: {r_mean:.4f} [{r_lo:.4f}, {r_hi:.4f}]  p={p_val:.4f}")

    results = {
        'clean': {str(s): clean_results[s] for s in clean_results},
        'corrupted': {str(s): corrupted_results[s] for s in corrupted_results},
    }
    with open(output_dir / 'cifar10_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
