"""
CIFAR-10 replication: runs all three generalization claims.
  1. CKA localization (ReLU layers vs linear layers)
  2. ROME detection (targeted recovery vs random baseline)
  3. Rank-ablation gap (clean vs corrupted at fc3)
"""
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.model import CIFAR10MLP
from models.cifarnet import CIFARNet
from utils.metrics import linear_cka, evaluate_class_accuracy
from utils.stats import compute_ci, paired_t_test
from analysis.rank_ablation import run_rank_ablation, _get_weight, _set_weight

CORRUPTION_CONFIGS = [
    {'source': 7, 'target': 1, 'label': '7→1'},
    {'source': 1, 'target': 7, 'label': '1→7'},
]


def get_cifar10_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    dataset = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_activations(model, dataloader, device, max_samples=5000):
    model.eval()
    all_layers = {k: [] for k in ['input', 'fc1_pre_activation', 'fc1_post_activation',
                                   'fc2_pre_activation', 'fc2_post_activation', 'output']}
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            out = model.forward_with_all_layers(data)
            for k in all_layers:
                all_layers[k].append(out[k].cpu())
            if sum(x.size(0) for x in all_layers['input']) >= max_samples:
                break
    return {k: torch.cat(v)[:max_samples] for k, v in all_layers.items()}


def run_cka_replication(clean_models, corrupted_models, loader, device):
    """Replication target 1: CKA localization."""
    print("\n=== CIFAR-10 Replication: CKA Localization ===")
    cka_pairs = ['input_fc1_pre', 'fc1_pre_fc1_post', 'fc1_post_fc2_pre',
                 'fc2_pre_fc2_post', 'fc2_post_output']
    pair_map = {
        'input_fc1_pre': ('input', 'fc1_pre_activation'),
        'fc1_pre_fc1_post': ('fc1_pre_activation', 'fc1_post_activation'),
        'fc1_post_fc2_pre': ('fc1_post_activation', 'fc2_pre_activation'),
        'fc2_pre_fc2_post': ('fc2_pre_activation', 'fc2_post_activation'),
        'fc2_post_output': ('fc2_post_activation', 'output'),
    }
    results = {}
    for pair_name in cka_pairs:
        k1, k2 = pair_map[pair_name]
        clean_vals = []
        corr_vals = []
        for seed in clean_models:
            acts = get_activations(clean_models[seed], loader, device)
            clean_vals.append(linear_cka(acts[k1], acts[k2]))
        for seed in corrupted_models:
            acts = get_activations(corrupted_models[seed], loader, device)
            corr_vals.append(linear_cka(acts[k1], acts[k2]))
        if clean_vals and corr_vals:
            c_mean, c_lo, c_hi = compute_ci(clean_vals)
            r_mean, r_lo, r_hi = compute_ci(corr_vals)
            delta = c_mean - r_mean
            _, p_val = paired_t_test(clean_vals, corr_vals)
            print(f"  {pair_name:25s}  Clean: {c_mean:.4f} [{c_lo:.4f}, {c_hi:.4f}]  "
                  f"Corrupted: {r_mean:.4f} [{r_lo:.4f}, {r_hi:.4f}]  "
                  f"Δ={delta:+.4f}  p={p_val:.4f}")
            results[pair_name] = {'clean_mean': c_mean, 'corrupted_mean': r_mean, 'delta': delta, 'p': p_val}
    return results


def run_rome_replication(clean_models, corrupted_models, loader, device):
    """Replication target 2: ROME detection."""
    print("\n=== CIFAR-10 Replication: ROME Detection ===")
    results = {}
    for cfg in CORRUPTION_CONFIGS:
        src, tgt, label = cfg['source'], cfg['target'], cfg['label']
        clean_recs = []
        corr_recs = []
        for seed in corrupted_models:
            model = corrupted_models[seed]
            baseline = evaluate_class_accuracy(model, loader, src, device)
            delta, u, v, _ = _compute_rome_edit_cifar(model, loader, device, tgt, 'fc3')
            if delta is None:
                continue
            W_orig = model.fc3.weight.data.clone()
            model.fc3.weight.data += delta
            post = evaluate_class_accuracy(model, loader, src, device)
            model.fc3.weight.data.copy_(W_orig)
            corr_recs.append(post - baseline)
        for seed in clean_models:
            model = clean_models[seed]
            baseline = evaluate_class_accuracy(model, loader, src, device)
            delta, u, v, _ = _compute_rome_edit_cifar(model, loader, device, tgt, 'fc3')
            if delta is None:
                continue
            W_orig = model.fc3.weight.data.clone()
            model.fc3.weight.data += delta
            post = evaluate_class_accuracy(model, loader, src, device)
            model.fc3.weight.data.copy_(W_orig)
            clean_recs.append(post - baseline)
        if clean_recs and corr_recs:
            cm, clo, chi = compute_ci(clean_recs)
            rm, rlo, rhi = compute_ci(corr_recs)
            _, p_val = paired_t_test(corr_recs, clean_recs)
            print(f"  {label:>8}  Clean: {cm:+.4f} [{clo:+.4f}, {chi:+.4f}]  "
                  f"Corrupted: {rm:+.4f} [{rlo:+.4f}, {rhi:+.4f}]  p={p_val:.4f}")
            results[label] = {'clean_recovery': cm, 'corrupted_recovery': rm, 'p': p_val}
    return results


def _compute_rome_edit_cifar(model, dataloader, device, target_class, layer='fc3'):
    """ROME rank-one edit for CIFAR10MLP."""
    model.eval()
    target_key = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            out = model.forward_with_all_layers(data)
            key = out['fc2_post_activation']
            mask = target == target_class
            if mask.any():
                target_key.append(key[mask].cpu())
    if not target_key:
        return None, None, None, layer
    u = torch.cat(target_key).mean(0).to(device)
    v = torch.zeros(model.output_dim, device=device)
    v[target_class] = 1.0
    W = model.fc3.weight.data
    delta = torch.outer(v - W @ u, u) / (u @ u + 1e-8)
    return delta, u, v, layer


def run_rank_ablation_replication(clean_models, corrupted_models, loader, device):
    """Replication target 3: Rank-ablation gap on fc3."""
    print("\n=== CIFAR-10 Replication: Rank Ablation (fc3) ===")
    criterion = nn.CrossEntropyLoss()
    results = {}
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        clean_accs = []
        corr_accs = []
        for seed in clean_models:
            model = clean_models[seed]
            W_orig = _get_weight(model, 'fc3').clone()
            U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
            k_eff = min(k, len(S))
            W_k = (U[:, :k_eff] * S[:k_eff]) @ Vh[:k_eff, :]
            _set_weight(model, 'fc3', W_k)
            correct = total = 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += len(y)
            clean_accs.append(100. * correct / total)
            _set_weight(model, 'fc3', W_orig)
        for seed in corrupted_models:
            model = corrupted_models[seed]
            W_orig = _get_weight(model, 'fc3').clone()
            U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
            k_eff = min(k, len(S))
            W_k = (U[:, :k_eff] * S[:k_eff]) @ Vh[:k_eff, :]
            _set_weight(model, 'fc3', W_k)
            correct = total = 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += len(y)
            corr_accs.append(100. * correct / total)
            _set_weight(model, 'fc3', W_orig)
        if clean_accs and corr_accs:
            cm, clo, chi = compute_ci(clean_accs)
            rm, rlo, rhi = compute_ci(corr_accs)
            gap = cm - rm
            print(f"  Rank {k:2d}:  Clean={cm:.2f}% [{clo:.2f}, {chi:.2f}]  "
                  f"Corrupted={rm:.2f}% [{rlo:.2f}, {rhi:.2f}]  Gap={gap:.2f}pp")
            results[k] = {'clean_acc': cm, 'corrupted_acc': rm, 'gap': gap}
    return results


def load_model(ckpt_path: str, model_type: str, device):
    if model_type == 'cifar10mlp':
        return CIFAR10MLP.load_checkpoint(ckpt_path, device)
    elif model_type == 'cifarnet':
        return CIFARNet.load_checkpoint(ckpt_path, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 replication of all three claims')
    parser.add_argument('--clean-dir', type=str, default='outputs/cifar10/clean')
    parser.add_argument('--corrupted-dir', type=str, default='outputs/cifar10/corrupted')
    parser.add_argument('--output-dir', type=str, default='outputs/cifar10/replication')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])
    parser.add_argument('--model-type', type=str, default='cifar10mlp', choices=['cifar10mlp', 'cifarnet'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_cifar10_loader()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_models = {}
    corrupted_models = {}
    for seed in args.seeds:
        ckpt = Path(args.clean_dir) / f"seed_{seed}" / 'final_model.pt'
        if ckpt.exists():
            clean_models[seed] = load_model(str(ckpt), args.model_type, device)
        ckpt = Path(args.corrupted_dir) / f"noise_0.2" / f"seed_{seed}" / 'final_model.pt'
        if ckpt.exists():
            corrupted_models[seed] = load_model(str(ckpt), args.model_type, device)

    results = {}
    results['cka'] = run_cka_replication(clean_models, corrupted_models, loader, device)
    results['rome'] = run_rome_replication(clean_models, corrupted_models, loader, device)
    results['rank_ablation'] = run_rank_ablation_replication(clean_models, corrupted_models, loader, device)

    with open(output_dir / 'cifar_replication_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
