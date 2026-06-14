"""
Phase 3: Influence functions and memorization metrics.
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.model import MNISTNet
from utils.stats import compute_ci


def compute_hessian_vector_product(model, data, target, vector, criterion):
    """Compute HVP: H * v using Pearlmutter's trick."""
    model.train()
    
    # First gradient
    output = model(data)
    loss = criterion(output, target)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    
    # Gradient-vector product
    gv = sum((g * v).sum() for g, v in zip(grads, vector) if g is not None)
    
    # Second gradient (HVP)
    hvp = torch.autograd.grad(gv, model.parameters(), retain_graph=False)
    
    return hvp


def conjugate_gradient(model, data, target, b, criterion, max_iter=50, tol=1e-6):
    """Solve Hx = b using conjugate gradient."""
    x = [torch.zeros_like(p) for p in model.parameters()]
    r = [b_i.clone() for b_i in b]
    p = [r_i.clone() for r_i in r]
    rsold = sum((r_i * r_i).sum() for r_i in r)
    
    for _ in range(max_iter):
        hvp = compute_hessian_vector_product(model, data, target, p, criterion)
        
        alpha = rsold / sum((p_i * hvp_i).sum() for p_i, hvp_i in zip(p, hvp) if hvp_i is not None)
        
        x = [x_i + alpha * p_i for x_i, p_i in zip(x, p)]
        r = [r_i - alpha * hvp_i for r_i, hvp_i in zip(r, hvp) if hvp_i is not None]
        
        rsnew = sum((r_i * r_i).sum() for r_i in r)
        if rsnew < tol:
            break
        
        beta = rsnew / rsold
        p = [r_i + beta * p_i for r_i, p_i in zip(r, p)]
        rsold = rsnew
    
    return x


def compute_influence(model, train_loader, test_loader, criterion, device, 
                     max_train_samples=1000, max_test_samples=100):
    """Compute influence of training points on test loss."""
    model.eval()
    
    # Get test gradient (average over test samples)
    test_grads = None
    test_count = 0
    
    for data, target in test_loader:
        if test_count >= max_test_samples:
            break
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        grads = torch.autograd.grad(loss, model.parameters())
        
        if test_grads is None:
            test_grads = [g.clone() for g in grads]
        else:
            test_grads = [tg + g for tg, g in zip(test_grads, grads)]
        test_count += data.size(0)
    
    test_grads = [g / test_count for g in test_grads]
    
    # Compute influence for each training sample
    influences = []
    
    for data, target in train_loader:
        if len(influences) >= max_train_samples:
            break
        data, target = data.to(device), target.to(device)
        
        # Compute training gradient for this sample
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_grads = torch.autograd.grad(loss, model.parameters())
        
        # Solve H^-1 * test_grad
        h_inv_test_grad = conjugate_gradient(model, data, target, test_grads, criterion)
        
        # Influence = - train_grad^T * H^-1 * test_grad
        influence = -sum((tg * hg).sum() for tg, hg in zip(train_grads, h_inv_test_grad))
        influences.append(influence.item())
    
    return np.array(influences)


def compute_memorization_score(model, train_loader, device,
                               corrupted_indices=None, n_train=60000):
    """
    Compute per-sample memorization via loss gap.

    Uses ground-truth corrupted_indices when available (non-circular),
    falls back to loss-quantile otherwise (exploratory / clean-model use).

    Returns dict with per_sample_loss, memorized_mask, loss_gap,
    and definition ('ground_truth' or 'loss_quantile').
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Track sample indices to align with corrupted_indices
    all_losses = []
    all_correct = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1)

            all_losses.extend(loss.cpu().numpy())
            all_correct.extend(pred.eq(target).cpu().numpy())
            # Compute global indices for this batch
            start = batch_idx * train_loader.batch_size
            batch_indices = list(range(start, start + len(data)))
            all_indices.extend(batch_indices)

    losses = np.array(all_losses)
    correct = np.array(all_correct)

    if corrupted_indices is not None:
        all_indices = np.array(all_indices)
        memorized_mask = np.isin(all_indices, corrupted_indices)
        forgotten_mask = ~memorized_mask
        definition = 'ground_truth'
    else:
        memorized_mask = correct & (losses < np.percentile(losses, 25))
        forgotten_mask = ~correct | (losses > np.percentile(losses, 75))
        definition = 'loss_quantile'

    mem_losses = losses[memorized_mask]
    forg_losses = losses[forgotten_mask]
    loss_gap = (forg_losses.mean() - mem_losses.mean()
                if len(mem_losses) > 0 and len(forg_losses) > 0 else 0.0)

    return {
        'mean_loss': float(losses.mean()),
        'std_loss': float(losses.std()),
        'accuracy': float(correct.mean()),
        'memorized_fraction': float(memorized_mask.mean()),
        'forgotten_fraction': float(forgotten_mask.mean()),
        'loss_gap': float(loss_gap),
        'memorized_mean_loss': float(mem_losses.mean()) if len(mem_losses) > 0 else 0.0,
        'non_memorized_mean_loss': float(forg_losses.mean()) if len(forg_losses) > 0 else 0.0,
        'definition': definition
    }


def get_data_loaders(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Influence functions and memorization')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='outputs/analysis/phase3')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--max-train-samples', type=int, default=500)
    parser.add_argument('--max-test-samples', type=int, default=50)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders(config['training']['batch_size'], config['training']['num_workers'])
    criterion = nn.CrossEntropyLoss()
    
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
        
        # Try to load ground truth corruption indices
        corrupt_path = Path(args.checkpoint_dir) / f"seed_{seed}" / 'corrupt_indices.npy'
        corrupted_indices = None
        if corrupt_path.exists():
            corrupted_indices = np.load(corrupt_path)
            print(f"  Loaded {len(corrupted_indices)} ground-truth corrupted indices")

        # Memorization metrics
        memo_results = compute_memorization_score(model, train_loader, device,
                                                   corrupted_indices=corrupted_indices)
        
        all_results[seed] = {'memorization': memo_results}
        
        print(f"Seed {seed}: Acc={memo_results['accuracy']:.4f}, "
              f"Memorized={memo_results['memorized_fraction']:.4f}, "
              f"Forgotten={memo_results['forgotten_fraction']:.4f}, "
              f"LossGap={memo_results['loss_gap']:.4f}")
    
    # Aggregate
    print("\n=== Aggregated Memorization Results ===")
    for metric in ['accuracy', 'memorized_fraction', 'forgotten_fraction', 'loss_gap', 'mean_loss']:
        values = [all_results[s]['memorization'][metric] for s in args.seeds if all_results[s]]
        if values:
            mean, ci_low, ci_high = compute_ci(values)
            print(f"{metric}: {mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    
    with open(output_dir / 'phase3_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()