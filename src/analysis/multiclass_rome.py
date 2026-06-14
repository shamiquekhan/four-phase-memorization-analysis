"""
Multi-class ROME validation using pre-trained corrupted checkpoints.
Validates ROME recovery across all classes using existing label-noise models.
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
from models.model import MNISTNet
from utils.stats import compute_ci
from utils.metrics import evaluate_class_accuracy


CORRUPTION_CONFIGS = [
    {'source': 7, 'target': 1, 'label': '7→1'},
    {'source': 1, 'target': 7, 'label': '1→7'},
    {'source': 5, 'target': 6, 'label': '5→6'},
    {'source': 0, 'target': 8, 'label': '0→8'},
]


def _get_hidden_activations(model, data, device):
    """Extract hidden (fc1_post) activations for ROME on fc2."""
    model.eval()
    with torch.no_grad():
        out = model.forward_with_all_layers(data)
        return out['fc1_post_activation']


def _get_input_activations(model, data, device):
    """Extract flattened input activations for ROME on fc1."""
    with torch.no_grad():
        return data.view(data.size(0), -1)


def compute_rome_edit(model, dataloader, device, target_class, layer='fc2'):
    """Compute rank-1 ROME edit for a specified layer.

    For fc2: u = mean hidden activation, v = one-hot target output.
    For fc1: u = mean input activation, v = mean target-class hidden
             activation (shifts source inputs toward target-like hidden reps).

    Returns:
        (delta, u, v, layer)
    """
    model.eval()

    if layer == 'fc2':
        target_key = []
        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(device)
                key = _get_hidden_activations(model, data, device)
                mask = target == target_class
                if mask.any():
                    target_key.append(key[mask].cpu())

        if not target_key:
            return None, None, None, layer

        u = torch.cat(target_key).mean(0).to(device)
        v = torch.zeros(model.output_dim, device=device)
        v[target_class] = 1.0
        W = model.fc2.weight.data

    elif layer == 'fc1':
        target_inputs, target_hiddens = [], []
        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(device)
                inp = _get_input_activations(model, data, device)
                hidden = _get_hidden_activations(model, data, device)
                mask = target == target_class
                if mask.any():
                    target_inputs.append(inp[mask].cpu())
                    target_hiddens.append(hidden[mask].cpu())

        if not target_inputs:
            return None, None, None, layer

        u = torch.cat(target_inputs).mean(0).to(device)
        v = torch.cat(target_hiddens).mean(0).to(device)
        W = model.fc1.weight.data

    else:
        raise ValueError(f"Unknown layer: {layer}")

    # ROME rank-one update (Meng et al., 2022)
    delta = torch.outer(v - W @ u, u) / (u @ u + 1e-8)
    return delta, u, v, layer


def apply_rome_edit(model, delta, layer='fc2'):
    """Apply precomputed ROME edit to specified layer."""
    with torch.no_grad():
        if layer == 'fc2':
            model.fc2.weight.data += delta
        elif layer == 'fc1':
            model.fc1.weight.data += delta


def find_broken_class(model, test_loader, device):
    """Find the class with lowest accuracy in corrupted model."""
    accs = {}
    for c in range(10):
        accs[c] = evaluate_class_accuracy(model, test_loader, c, device)
    worst = min(accs, key=accs.get)
    return worst, accs


def run_rome_experiment(model, test_loader, device, target_class, layer='fc2'):
    """Run ROME on a specified layer and return recovery, side effects."""
    pre_accs = {c: evaluate_class_accuracy(model, test_loader, c, device) for c in range(10)}

    delta, u, v, used_layer = compute_rome_edit(model, test_loader, device, target_class, layer)
    if delta is None:
        return 0.0, 0.0, 0.0, pre_accs, {}

    weight_attr = 'fc2.weight' if used_layer == 'fc2' else 'fc1.weight'
    W_orig = getattr(model, weight_attr.split('.')[0]).weight.data.clone()
    apply_rome_edit(model, delta, used_layer)

    post_accs = {c: evaluate_class_accuracy(model, test_loader, c, device) for c in range(10)}

    recovery = post_accs[target_class] - pre_accs[target_class]
    other_classes = [c for c in range(10) if c != target_class]
    side_effects = float(np.mean([abs(post_accs[c] - pre_accs[c]) for c in other_classes]))
    magnitude = (getattr(model, weight_attr.split('.')[0]).weight.data - W_orig).norm(p='fro').item()

    getattr(model, weight_attr.split('.')[0]).weight.data.copy_(W_orig)

    return recovery, side_effects, magnitude, pre_accs, {'layer': used_layer, 'delta_norm': delta.norm().item()}


def run_multi_layer_rome(model, test_loader, device, target_class):
    """Apply ROME to fc2, then fc1 sequentially; compare to single-layer."""
    results = {}

    # fc2-only
    r_fc2, s_fc2, m_fc2, pre, meta2 = run_rome_experiment(model, test_loader, device, target_class, 'fc2')
    results['fc2_only'] = {'recovery': r_fc2, 'side_effects': s_fc2, 'magnitude': m_fc2}

    # fc1-only
    r_fc1, s_fc1, m_fc1, pre, meta1 = run_rome_experiment(model, test_loader, device, target_class, 'fc1')
    results['fc1_only'] = {'recovery': r_fc1, 'side_effects': s_fc1, 'magnitude': m_fc1}

    # Both layers (fc2 then fc1 sequentially)
    delta2, u2, v2, _ = compute_rome_edit(model, test_loader, device, target_class, 'fc2')
    W2_orig = model.fc2.weight.data.clone()
    apply_rome_edit(model, delta2, 'fc2')
    delta1, u1, v1, _ = compute_rome_edit(model, test_loader, device, target_class, 'fc1')
    W1_orig = model.fc1.weight.data.clone()
    apply_rome_edit(model, delta1, 'fc1')
    post_both = {c: evaluate_class_accuracy(model, test_loader, c, device) for c in range(10)}
    model.fc1.weight.data.copy_(W1_orig)
    model.fc2.weight.data.copy_(W2_orig)
    results['both_layers'] = {
        'recovery': post_both[target_class] - pre[target_class],
        'side_effects': float(np.mean([abs(post_both[c] - pre[c]) for c in range(10) if c != target_class])),
        'magnitude': (delta2.norm().item() + delta1.norm().item())
    }

    return results


def get_test_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    parser = argparse.ArgumentParser(description='Multi-class ROME validation')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/targeted_corrupted')
    parser.add_argument('--output-dir', type=str, default='outputs/analysis/multiclass_rome')
    parser.add_argument('--clean-dir', type=str, default='outputs/clean')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(3)))
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_loader = get_test_loader(config['training']['batch_size'])
    corruption_configs = config.get('phase4', {}).get('corruption_configs', CORRUPTION_CONFIGS)

    # Run ROME on each corruption config
    all_results = {}
    multi_results = {}
    for cfg in corruption_configs:
        label = cfg['label']
        target = cfg['source']
        print(f"\n=== Corruption config: {label} ===")
        all_results[label] = {'recovery': [], 'side_effects': [], 'magnitude': [],
                              'pre_accs': [], 'post_accs': []}
        multi_results[label] = {'fc2_only': {'recovery': [], 'side_effects': [], 'magnitude': []},
                                'fc1_only': {'recovery': [], 'side_effects': [], 'magnitude': []},
                                'both_layers': {'recovery': [], 'side_effects': [], 'magnitude': []}}

        for seed in args.seeds:
            ckpt_path = Path(args.checkpoint_dir) / f"src{cfg['source']}_tgt{cfg['target']}" / f"seed_{seed}" / 'final_model.pt'
            if not ckpt_path.exists():
                print(f"  Checkpoint not found: {ckpt_path}")
                continue

            model = MNISTNet(
                config['model']['input_dim'],
                config['model']['hidden_dim'],
                config['model']['output_dim']
            ).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # fc2-only (original experiment)
            recovery, side_effects, magnitude, pre_accs, _ = run_rome_experiment(
                model, test_loader, device, target, 'fc2'
            )
            all_results[label]['recovery'].append(recovery)
            all_results[label]['side_effects'].append(side_effects)
            all_results[label]['magnitude'].append(magnitude)

            # Multi-layer comparison (fc1-only, fc2-only, both)
            layer_results = run_multi_layer_rome(model, test_loader, device, target)
            for k in ['fc2_only', 'fc1_only', 'both_layers']:
                multi_results[label][k]['recovery'].append(layer_results[k]['recovery'])
                multi_results[label][k]['side_effects'].append(layer_results[k]['side_effects'])
                multi_results[label][k]['magnitude'].append(layer_results[k]['magnitude'])

            print(f"  seed={seed} fc2={layer_results['fc2_only']['recovery']:.4f} "
                  f"fc1={layer_results['fc1_only']['recovery']:.4f} "
                  f"both={layer_results['both_layers']['recovery']:.4f}")

    # Summary
    print("\n=== MULTI-CLASS ROME SUMMARY (fc2-only) ===")
    header = f"{'Config':>8} {'Recovery':>18} {'Side Effects':>16} {'Magnitude':>12}"
    print(header)
    print('-' * len(header))
    for label in all_results:
        r_mean, r_lo, r_hi = compute_ci(all_results[label]['recovery'])
        s_mean, s_lo, s_hi = compute_ci(all_results[label]['side_effects'])
        m_mean, m_lo, m_hi = compute_ci(all_results[label]['magnitude'])
        print(f"{label:>8}  {r_mean:+.4f} [{r_lo:+.4f}, {r_hi:+.4f}]  "
              f"{s_mean:.4f} [{s_lo:.4f}, {s_hi:.4f}]  "
              f"{m_mean:.4f} [{m_lo:.4f}, {m_hi:.4f}]")

    print("\n=== MULTI-LAYER ROME COMPARISON ===")
    for k in ['fc2_only', 'fc1_only', 'both_layers']:
        print(f"\n  {k}:")
        hl = f"{'Config':>8} {'Recovery':>18} {'Side Effects':>16} {'Magnitude':>12}"
        print(f"    {hl}")
        print(f"    {'-' * len(hl)}")
        for label in multi_results:
            r_mean, r_lo, r_hi = compute_ci(multi_results[label][k]['recovery'])
            s_mean, s_lo, s_hi = compute_ci(multi_results[label][k]['side_effects'])
            m_mean, m_lo, m_hi = compute_ci(multi_results[label][k]['magnitude'])
            print(f"    {label:>8}  {r_mean:+.4f} [{r_lo:+.4f}, {r_hi:+.4f}]  "
                  f"{s_mean:.4f} [{s_lo:.4f}, {s_hi:.4f}]  "
                  f"{m_mean:.4f} [{m_lo:.4f}, {m_hi:.4f}]")

    with open(output_dir / 'multiclass_rome_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(output_dir / 'multilayer_rome_comparison.json', 'w') as f:
        json.dump(multi_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()