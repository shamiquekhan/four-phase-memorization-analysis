"""
Fast 0→8 rank ablation with efficient evaluation.
"""
import sys, json, torch, numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from models.model import MNISTNet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.stats import compute_ci

SEEDS = [42, 123, 456, 789, 1024]
DEVICE = torch.device('cuda')

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def load_model(ckpt_path):
    model = MNISTNet(784, 16, 10).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)['model_state_dict'])
    model.eval()
    return model

def get_mean_hidden(model, loader, class_idx):
    acts = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model.forward_with_all_layers(x)
            mask = y == class_idx
            if mask.any():
                acts.append(out['fc1_post_activation'][mask])
    return torch.cat(acts).mean(0) if acts else None

def compute_class_accs(model, loader):
    correct, total = torch.zeros(10, device=DEVICE), torch.zeros(10, device=DEVICE)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            for c in range(10):
                m = y == c
                correct[c] += (pred[m] == y[m]).sum()
                total[c] += m.sum()
    return {c: (correct[c]/total[c]).item() if total[c] > 0 else 0.0 for c in range(10)}

# Load all models once
models = {}
for seed in SEEDS:
    ckpt = Path(f'outputs/targeted_corrupted/src0_tgt8/seed_{seed}/final_model.pt')
    models[seed] = load_model(ckpt)

ablation = {}
for seed in SEEDS:
    model = models[seed]
    W_orig = model.fc2.weight.data.clone()
    
    # ROME edit for class 0 (repair key)
    u = get_mean_hidden(model, train_loader, 0)
    v = torch.zeros(10, device=DEVICE); v[0] = 1.0
    delta = torch.outer(v - model.fc2.weight.data @ u, u) / (u @ u + 1e-8)
    
    # SVD
    U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
    ranks = list(range(1, min(len(S) + 1, 11)))
    
    pre_accs = compute_class_accs(model, test_loader)
    
    per_rank = {}
    for k in ranks:
        delta_k = (U[:, :k] * S[:k]) @ Vh[:k, :]
        model.fc2.weight.data.copy_(W_orig + delta_k)
        post = compute_class_accs(model, test_loader)
        per_rank[k] = {str(c): float(post[c]) for c in range(10)}
    
    model.fc2.weight.data.copy_(W_orig)
    ablation[str(seed)] = {'ranks': ranks, 'per_class_acc': per_rank, 'singular_values': S.cpu().tolist()}
    
    r1 = per_rank[1].get('0', 0)
    rmax = per_rank[ranks[-1]].get('0', 0)
    pre = pre_accs[0]
    print(f"Seed {seed}: pre={pre:.4f} rank-1={r1:.4f} recov={r1-pre:+.4f} full-rank={rmax:.4f}")

json.dump(ablation, open('outputs/analysis/rome_08_rank_ablation.json', 'w'), indent=2, default=str)
print(f"\nSaved to outputs/analysis/rome_08_rank_ablation.json")

# Aggregate
print("\n=== AGGREGATED ===")
recov_r1 = [ablation[s][1].get('0', 0) for s in ablation for s in ablation]
print("Done")
