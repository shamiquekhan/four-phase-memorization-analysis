"""
Phase 4: ROME (Rank-One Model Editing) analysis for memorization localization.
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


def _get_layer_activations(model, data, layer_name):
    """Get activations for a specific layer."""
    acts = model.forward_with_all_layers(data)
    if layer_name == 'fc1':
        return acts['fc1_post_activation']
    elif layer_name == 'fc2':
        return acts['output']
    else:
        raise ValueError(f"Unknown layer: {layer_name}")


def compute_rome_edit(model, dataloader, device, target_class, layer_name='fc1'):
    """
    Compute ROME edit: find rank-1 update to change prediction for target class.
    Based on ROME: Rank-One Model Editing (Meng et al., 2022)
    """
    model.eval()
    
    # Get activations for target class
    target_acts = []
    other_acts = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            acts = _get_layer_activations(model, data, layer_name)
            
            mask = target == target_class
            if mask.any():
                target_acts.append(acts[mask])
            if (~mask).any():
                other_acts.append(acts[~mask])
    
    if not target_acts or not other_acts:
        return None
    
    target_acts = torch.cat(target_acts)
    other_acts = torch.cat(other_acts)
    
    # Mean activations
    mean_target = target_acts.mean(0)
    mean_other = other_acts.mean(0)
    
    # Difference vector (direction to change)
    delta = mean_target - mean_other
    
    # Get weight matrix for target layer
    if layer_name == 'fc1':
        W = model.fc1.weight.data
    elif layer_name == 'fc2':
        W = model.fc2.weight.data
    else:
        raise ValueError(f"Unknown layer: {layer_name}")
    
    # Rank-1 ROME update: project delta onto row space of W
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    w_target = W[target_class]
    v = delta / (delta.norm() + 1e-8)
    u = w_target / (w_target.norm() + 1e-8)
    
    # Rank-1 update
    rank1_update = torch.outer(u, v)
    
    # Effect on target class logit
    effect = (rank1_update @ mean_target).sum().item()
    
    return {
        'delta_norm': delta.norm().item(),
        'effect_on_target': effect,
        'rank1_update_norm': rank1_update.norm().item(),
        'target_class': target_class
    }


def compute_rome_for_all_classes(model, dataloader, device, layer_name='fc1'):
    """Compute ROME edits for all classes."""
    results = {}
    for c in range(10):
        edit = compute_rome_edit(model, dataloader, device, c, layer_name)
        if edit:
            results[c] = edit
    return results


def get_data_loader(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def main():
    parser = argparse.ArgumentParser(description='Phase 4: ROME analysis')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='outputs/analysis/phase4')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_data_loader(config['training']['batch_size'], config['training']['num_workers'])
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {seed: {} for seed in args.seeds}
    
    for seed in args.seeds:
        checkpoint_path = Path(args.checkpoint_dir) / f"seed_{seed}" / 'final_model.pt'
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found for seed {seed}")
            continue
        
        model = MNISTNet(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            output_dim=config['model']['output_dim'],
            activation=config['model']['activation']
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # ROME on fc1 (hidden layer)
        fc1_results = compute_rome_for_all_classes(model, train_loader, device, 'fc1')
        # ROME on fc2 (output layer)
        fc2_results = compute_rome_for_all_classes(model, train_loader, device, 'fc2')
        
        all_results[seed] = {
            'fc1': fc1_results,
            'fc2': fc2_results
        }
        
        # Summary
        fc1_effects = [fc1_results[c]['effect_on_target'] for c in fc1_results]
        fc2_effects = [fc2_results[c]['effect_on_target'] for c in fc2_results]
        print(f"Seed {seed}: FC1 mean effect={np.mean(fc1_effects):.4f}, FC2 mean effect={np.mean(fc2_effects):.4f}")
    
    # Aggregate
    print("\n=== Aggregated ROME Results ===")
    for layer in ['fc1', 'fc2']:
        effects_per_class = {c: [] for c in range(10)}
        for seed in args.seeds:
            if all_results[seed]:
                for c in all_results[seed][layer]:
                    effects_per_class[c].append(all_results[seed][layer][c]['effect_on_target'])
        
        for c in range(10):
            if effects_per_class[c]:
                mean, ci_low, ci_high = compute_ci(effects_per_class[c])
                print(f"{layer} Class {c}: {mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    
    with open(output_dir / 'phase4_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()