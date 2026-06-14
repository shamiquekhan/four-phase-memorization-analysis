"""
Phase 2: Representation similarity analysis - CKA, PCA, activation statistics.
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
from utils.metrics import linear_cka
from utils.stats import compute_ci, paired_t_test


def get_activations(model, dataloader, device, max_samples=5000):
    """Extract activations from all layers."""
    model.eval()
    all_input = []
    all_fc1_pre = []
    all_fc1_post = []
    all_output = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            out = model.forward_with_all_layers(data)
            
            all_input.append(out['input'].cpu())
            all_fc1_pre.append(out['fc1_pre_activation'].cpu())
            all_fc1_post.append(out['fc1_post_activation'].cpu())
            all_output.append(out['output'].cpu())
            
            if sum(x.size(0) for x in all_input) >= max_samples:
                break
    
    return {
        'input': torch.cat(all_input)[:max_samples],
        'fc1_pre': torch.cat(all_fc1_pre)[:max_samples],
        'fc1_post': torch.cat(all_fc1_post)[:max_samples],
        'output': torch.cat(all_output)[:max_samples]
    }


def compute_pca(activations, n_components=50):
    """Compute PCA on activations."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_components, activations.shape[1]))
    transformed = pca.fit_transform(activations.numpy())
    return transformed, pca.explained_variance_ratio_


def analyze_representations(model, dataloader, device):
    """Analyze representation similarity and structure."""
    acts = get_activations(model, dataloader, device)
    
    # CKA between layers
    cka_input_fc1pre = linear_cka(acts['input'], acts['fc1_pre'])
    cka_fc1pre_fc1post = linear_cka(acts['fc1_pre'], acts['fc1_post'])
    cka_fc1post_output = linear_cka(acts['fc1_post'], acts['output'])
    
    # Activation statistics
    stats = {}
    for name, act in acts.items():
        stats[name] = {
            'mean': act.mean().item(),
            'std': act.std().item(),
            'sparsity': (act == 0).float().mean().item(),
            'max': act.max().item(),
            'min': act.min().item()
        }
    
    # PCA on hidden layer
    pca_transformed, exp_var = compute_pca(acts['fc1_post'])
    
    return {
        'cka': {
            'input_fc1_pre': cka_input_fc1pre.item(),
            'fc1_pre_fc1_post': cka_fc1pre_fc1post.item(),
            'fc1_post_output': cka_fc1post_output.item()
        },
        'activation_stats': stats,
        'pca_explained_variance': exp_var.tolist(),
        'pca_cumulative_variance': np.cumsum(exp_var).tolist()
    }


def get_data_loader(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Representation similarity analysis')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--corrupted-checkpoint-dir', type=str, default=None,
                       help='If provided, runs paired t-test between clean and corrupted CKA values')
    parser.add_argument('--output-dir', type=str, default='outputs/analysis/phase2')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = get_data_loader(config['training']['batch_size'], config['training']['num_workers'])
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_phase2(checkpoint_dir, label):
        results = {seed: {} for seed in args.seeds}
        for seed in args.seeds:
            ckpt_path = Path(checkpoint_dir) / f"seed_{seed}" / 'final_model.pt'
            if not ckpt_path.exists():
                print(f"{label}: checkpoint not found for seed {seed}")
                continue
            model = MNISTNet(
                input_dim=config['model']['input_dim'],
                hidden_dim=config['model']['hidden_dim'],
                output_dim=config['model']['output_dim'],
                activation=config['model']['activation']
            ).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            results[seed] = analyze_representations(model, test_loader, device)
            res = results[seed]
            print(f"{label} seed {seed}: CKA(input,fc1_pre)={res['cka']['input_fc1_pre']:.4f}, "
                  f"CKA(fc1_pre,fc1_post)={res['cka']['fc1_pre_fc1_post']:.4f}, "
                  f"CKA(fc1_post,output)={res['cka']['fc1_post_output']:.4f}")
        return results
    
    all_results = run_phase2(args.checkpoint_dir, "clean")
    
    print("\n=== Aggregated CKA Results ===")
    for cka_key in ['input_fc1_pre', 'fc1_pre_fc1_post', 'fc1_post_output']:
        values = [all_results[s]['cka'][cka_key] for s in args.seeds if all_results[s]]
        if values:
            mean, ci_low, ci_high = compute_ci(values)
            print(f"CKA {cka_key}: {mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    
    if args.corrupted_checkpoint_dir:
        corr_results = run_phase2(args.corrupted_checkpoint_dir, "corrupted")
        print("\n=== Clean vs Corrupted Paired t-test ===")
        for cka_key in ['input_fc1_pre', 'fc1_pre_fc1_post', 'fc1_post_output']:
            clean_vals = [all_results[s]['cka'][cka_key] for s in args.seeds if all_results[s]]
            corr_vals = [corr_results[s]['cka'][cka_key] for s in args.seeds if corr_results[s]]
            if clean_vals and corr_vals and len(clean_vals) == len(corr_vals):
                c_mean, c_lo, c_hi = compute_ci(clean_vals)
                r_mean, r_lo, r_hi = compute_ci(corr_vals)
                t_stat, p_val = paired_t_test(clean_vals, corr_vals)
                print(f"CKA {cka_key}: Clean={c_mean:.4f} [{c_lo:.4f}, {c_hi:.4f}], "
                      f"Corrupted={r_mean:.4f} [{r_lo:.4f}, {r_hi:.4f}], "
                      f"Δ={c_mean - r_mean:+.4f}, t={t_stat:.3f}, p={p_val:.4f}")
        all_results = {'clean': all_results, 'corrupted': corr_results}
    
    with open(output_dir / 'phase2_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()