"""
Train CIFARNet on CIFAR-10 under clean and 20% label noise.
Mirrors train_clean.py and train_corrupted.py interface exactly.
"""
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.cifarnet import CIFARNet


def get_cifar10(train=True, noise_rate=0.0, seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    dataset = datasets.CIFAR10(root='data/', train=train, download=True,
                                transform=transform)
    corrupt_indices = []

    if train and noise_rate > 0:
        rng = np.random.RandomState(seed)
        n = len(dataset.targets)
        n_corrupt = int(n * noise_rate)
        corrupt_idx = rng.choice(n, n_corrupt, replace=False)
        corrupt_indices = corrupt_idx.tolist()
        for i in corrupt_idx:
            orig = dataset.targets[i]
            choices = [c for c in range(10) if c != orig]
            dataset.targets[i] = int(rng.choice(choices))

    return dataset, corrupt_indices


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data, corrupt_indices = get_cifar10(
        train=True, noise_rate=args.noise_rate, seed=args.seed
    )
    test_data, _ = get_cifar10(train=False)

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False)

    model = CIFARNet(hidden1=args.hidden1, hidden2=args.hidden2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.output_dir) / f'seed_{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / 'corrupt_indices.npy', np.array(corrupt_indices))

    best_test_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
        test_acc = correct / total

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model.save_checkpoint(
                out_dir / 'best_model.pt',
                metadata={'epoch': epoch, 'test_acc': test_acc,
                          'seed': args.seed, 'noise_rate': args.noise_rate}
            )

    print(f'Seed {args.seed} | Best test acc: {best_test_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-rate', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden1', type=int, default=256)
    parser.add_argument('--hidden2', type=int, default=128)
    parser.add_argument('--output-dir', type=str, default='outputs/cifar')
    args = parser.parse_args()
    train(args)
