"""Width-controlled CIFAR-10 training for scaling analysis (G2).
Trains CIFAR-10 MLP at various widths to measure monosemanticity scaling."""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.model import CIFAR10MLP


class CIFAR10ScaledMLP(nn.Module):
    """CIFAR-10 MLP with configurable hidden widths for scaling experiments."""
    def __init__(self, input_dim=3072, hidden_dim=256, output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def get_hidden(self, x):
        h = self.net[1](self.net[0](x))
        h = self.net[2](h)
        return self.net[3](h)

    def forward_with_all_layers(self, x):
        flat = self.net[0](x)
        fc1_pre = self.net[1](flat)
        fc1_post = self.net[2](fc1_pre)
        fc2_pre = self.net[3](fc1_post)
        fc2_post = self.net[4](fc2_pre)
        output = self.net[5](fc2_post)
        return {
            'input': x,
            'fc1_pre_activation': fc1_pre,
            'fc1_post_activation': fc1_post,
            'fc2_pre_activation': fc2_pre,
            'fc2_post_activation': fc2_post,
            'output': output
        }

    def save_checkpoint(self, path, metadata=None):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'architecture': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim
            }
        }
        if metadata:
            checkpoint['metadata'] = metadata
        torch.save(checkpoint, path)

    def get_weight_info(self):
        return {'total_params': sum(p.numel() for p in self.parameters())}


def get_data_loaders(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_dataset = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data, target in tqdm(loader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Evaluating", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 scaled MLP')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--output-dir', type=str, default='outputs/cifar10/scaling')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, hidden_dim={args.hidden_dim}")

    model = CIFAR10ScaledMLP(
        input_dim=3072,
        hidden_dim=args.hidden_dim,
        output_dim=10,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_data_loaders(
        batch_size=128, num_workers=config['training']['num_workers']
    )

    output_dir = Path(args.output_dir) / f"hidden_{args.hidden_dim}" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    epochs = config.get('scaling', {}).get('epochs', 100)

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            model.save_checkpoint(output_dir / 'best_model.pt', {'epoch': epoch, 'test_acc': test_acc})

    model.save_checkpoint(output_dir / 'final_model.pt', {'epoch': epochs, 'test_acc': test_acc})
    torch.save(history, output_dir / 'history.pt')
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
