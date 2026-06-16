"""
CIFAR-10 Phase 3: Influence functions and memorization metrics.
Replicates MNIST Phase 3 findings on CIFAR-10 with a 3-layer MLP.
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
from utils.stats import compute_ci


def compute_memorization_score(model, train_loader, device,
                               corrupted_indices=None, n_train=50000):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
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
            start = batch_idx * train_loader.batch_size
            batch_indices = list(range(start, start + len(data)))
            all_indices.extend(batch_indices)

    losses = np.array(all_losses)
    correct = np.array(all_correct)

    if corrupted_indices is not None:
        all_indices_arr = np.array(all_indices)
        memorized_mask = np.isin(all_indices_arr, corrupted_indices)
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


def compute_gradient_alignment(model, dataloader, corrupt_indices, device='cpu'):
    model.eval()
    genuine_grad = None
    corrupt_grad = None

    for batch_idx, (x, y) in enumerate(dataloader):
        idx = list(range(batch_idx * dataloader.batch_size,
                         batch_idx * dataloader.batch_size + len(x)))
        x, y = x.to(device), y.to(device)
        is_corrupt = torch.tensor([i in corrupt_indices for i in idx])

        for flag, mask in [(False, ~is_corrupt), (True, is_corrupt)]:
            if mask.sum() == 0:
                continue
            x_sub, y_sub = x[mask], y[mask]
            model.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x_sub), y_sub)
            loss.backward()
            # Use fc3 (output layer) gradients, matching CIFAR-10 architecture
            g = model.fc3.weight.grad.detach().clone().flatten()
            if flag:
                corrupt_grad = g if corrupt_grad is None else corrupt_grad + g
            else:
                genuine_grad = g if genuine_grad is None else genuine_grad + g

    if genuine_grad is None or corrupt_grad is None:
        return None

    cos_sim = torch.nn.functional.cosine_similarity(
        genuine_grad.unsqueeze(0),
        corrupt_grad.unsqueeze(0)
    ).item()

    return cos_sim


def get_data_loaders(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_dataset = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Phase 3: memorization metrics')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--clean-dir', type=str, default='outputs/cifar10/clean')
    parser.add_argument('--corrupted-dir', type=str, default='outputs/cifar10/corrupted/noise_0.2')
    parser.add_argument('--output-dir', type=str, default='outputs/cifar10/analysis/phase3')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders(
        config.get('cifar10', {}).get('batch_size', 128),
        4
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for seed in args.seeds:
        # Load clean model
        ckpt_clean = Path(args.clean_dir) / f"seed_{seed}" / 'final_model.pt'
        if ckpt_clean.exists():
            model = CIFAR10MLP.load_checkpoint(str(ckpt_clean), device)
            model.eval()
            memo_clean = compute_memorization_score(model, train_loader, device, corrupted_indices=None)
            print(f"Clean seed={seed}: Acc={memo_clean['accuracy']:.4f}, LossGap={memo_clean['loss_gap']:.4f}")
        else:
            print(f"Clean seed={seed}: checkpoint not found")
            continue

        # Load corrupted model
        ckpt_corr = Path(args.corrupted_dir) / f"seed_{seed}" / 'final_model.pt'
        corrupt_path = Path(args.corrupted_dir) / f"seed_{seed}" / 'corrupt_indices.npy'
        corrupted_indices = None
        if corrupt_path.exists():
            corrupted_indices = np.load(corrupt_path)

        if ckpt_corr.exists():
            model = CIFAR10MLP.load_checkpoint(str(ckpt_corr), device)
            model.eval()
            memo_corr = compute_memorization_score(model, train_loader, device,
                                                    corrupted_indices=corrupted_indices)

            grad_align = None
            if corrupted_indices is not None and len(corrupted_indices) > 0:
                grad_align = compute_gradient_alignment(model, train_loader, corrupted_indices, device)

            memo_corr['gradient_alignment'] = grad_align
            align_str = f", GradAlign={grad_align:.4f}" if grad_align is not None else ""
            print(f"Corrupted seed={seed}: Acc={memo_corr['accuracy']:.4f}, "
                  f"LossGap={memo_corr['loss_gap']:.4f}{align_str}")
        else:
            print(f"Corrupted seed={seed}: checkpoint not found")
            continue

        all_results[str(seed)] = {
            'clean': memo_clean,
            'corrupted': memo_corr,
        }

    if not all_results:
        print("No results obtained.")
        return

    print("\n=== CIFAR-10 Phase 3 Aggregated Results ===")
    for condition in ['clean', 'corrupted']:
        print(f"\n  {condition.upper()} model:")
        for metric in ['accuracy', 'mean_loss', 'loss_gap', 'gradient_alignment']:
            values = [all_results[s][condition][metric]
                      for s in all_results
                      if all_results[s][condition].get(metric) is not None]
            if values:
                mean, ci_lo, ci_hi = compute_ci(values)
                print(f"    {metric}: {mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    with open(output_dir / 'cifar10_phase3_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
