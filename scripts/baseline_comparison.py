"""
Baseline comparison for memorization detection.
Compares ROME delta-norm against:
1. Linear probe (logistic regression on hidden activations to predict corruption)
2. Spectral norm ratio (from Phase 1)

Usage: python scripts/baseline_comparison.py
"""
import sys, json, torch, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from models.model import MNISTNet
from analysis.phase4_rome import compute_rome_for_all_classes
from analysis.phase1_basic import compute_weight_norms
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

SEEDS = [42, 123, 456, 789, 1024]
NOISE_RATES = [0.1, 0.2, 0.4]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

def load_model(ckpt_path):
    model = MNISTNet(784, 16, 10).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)['model_state_dict'])
    model.eval()
    return model

def get_hidden_activations(model, loader):
    acts, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model.forward_with_all_layers(x)
            acts.append(out['fc1_post_activation'].cpu())
            labels.append(y)
    return torch.cat(acts).numpy(), torch.cat(labels).numpy()

print("=" * 60)
print("BASELINE COMPARISON: ROME vs LINEAR PROBE vs SPECTRAL RATIO")
print("=" * 60)

results = {}

for noise in NOISE_RATES:
    print(f"\n--- Noise rate: {noise} ---")
    
    # Load clean and corrupted models
    for label, seed_dir, is_clean in [("clean", "clean", True), (f"corrupted_{noise}", f"corrupted/noise_{noise}", False)]:
        rome_fc2 = {c: [] for c in range(10)}
        spectral_fc1, spectral_fc2 = [], []
        linear_probe_aucs = []
        
        for seed in SEEDS:
            if is_clean:
                ckpt = Path(f'outputs/clean/seed_{seed}/final_model.pt')
            else:
                ckpt = Path(f'outputs/corrupted/noise_{noise}/seed_{seed}/final_model.pt')
            
            if not ckpt.exists():
                continue
            
            model = load_model(ckpt)
            
            # 1. ROME delta-norm
            fc2 = compute_rome_for_all_classes(model, train_loader, DEVICE, 'fc2')
            for c in range(10):
                if c in fc2: rome_fc2[c].append(fc2[c]['delta_norm'])
            
            # 2. Spectral norms (from Phase 1)
            try:
                if is_clean:
                    p1_path = Path(f'outputs/analysis/phase1_clean/phase1_results.json')
                else:
                    p1_path = Path(f'outputs/analysis/phase1_corrupted/phase1_results.json')
                if p1_path.exists():
                    p1 = json.load(open(p1_path))
                    s = p1.get(str(seed), {})
                    if 'fc1_spectral_norm' in s:
                        spectral_fc1.append(s['fc1_spectral_norm'])
                    if 'fc2_spectral_norm' in s:
                        spectral_fc2.append(s['fc2_spectral_norm'])
            except Exception as e:
                print(f"    Warning: couldn't load spectral norms: {e}")
            
            # 3. Linear probe: train on hidden activations to detect corruption
            if not is_clean:
                try:
                    # Use hidden activations to predict whether a sample is corrupted
                    # Need clean and corrupted models' hidden activations
                    clean_ckpt = Path(f'outputs/clean/seed_{seed}/final_model.pt')
                    if clean_ckpt.exists():
                        clean_model = load_model(clean_ckpt)
                        H_clean, y_clean = get_hidden_activations(clean_model, train_loader)
                        H_corrupt, y_corrupt = get_hidden_activations(model, train_loader)
                        
                        # Load corruption indices
                        corrupt_indices = np.load(ckpt.parent.parent / f'seed_{seed}' / 'corrupt_indices.npy')
                        is_corrupted = np.zeros(len(train_dataset), dtype=int)
                        is_corrupted[corrupt_indices] = 1
                        
                        # Sample balanced subset
                        n_sample = 5000
                        idx = np.random.choice(len(train_dataset), n_sample, replace=False)
                        X = H_corrupt[idx]
                        y_binary = is_corrupted[idx]
                        
                        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
                        clf.fit(X, y_binary)
                        auc = roc_auc_score(y_binary, clf.predict_proba(X)[:, 1])
                        linear_probe_aucs.append(auc)
                except Exception as e:
                    print(f"    Linear probe failed for seed {seed}: {e}")

        # Report
        rome_avg = np.mean([np.mean(rome_fc2[c]) for c in range(10) if rome_fc2[c]])
        print(f"  {label}: ROME fc2 avg={rome_avg:.2f}")
        if spectral_fc2:
            print(f"  {label}: Spectral norm fc2={np.mean(spectral_fc2):.2f}")
        if linear_probe_aucs:
            print(f"  {label}: Linear probe AUC={np.mean(linear_probe_aucs):.3f}")
        
        results[label] = {
            'rome_fc2_avg': rome_avg,
            'rome_per_class': {str(c): float(np.mean(rome_fc2[c])) for c in range(10) if rome_fc2[c]},
            'spectral_fc2_mean': float(np.mean(spectral_fc2)) if spectral_fc2 else None,
            'linear_probe_auc': float(np.mean(linear_probe_aucs)) if linear_probe_aucs else None,
        }

# Cross-noise rate comparison
print("\n=== CROSS-NOISE COMPARISON ===")
print(f"{'Noise':>8} {'ROME fc2':>12} {'Spectral fc2':>14} {'Ratio':>10}")
print("-" * 50)
clean_rome = results.get('clean', {}).get('rome_fc2_avg', 0)
clean_spec = results.get('clean', {}).get('spectral_fc2_mean', 0)
for noise in NOISE_RATES:
    key = f'corrupted_{noise}'
    r = results.get(key, {})
    rome_val = r.get('rome_fc2_avg', 0)
    spec_val = r.get('spectral_fc2_mean', 0)
    rome_ratio = clean_rome / rome_val if rome_val else float('inf')
    spec_ratio = clean_spec / spec_val if spec_val else float('inf')
    print(f"  {noise:>4.1f}     {rome_val:>8.2f}      {spec_val:>8.2f}        {rome_ratio:.2f}x")

json.dump(results, open('outputs/analysis/baseline_comparison.json', 'w'), indent=2, default=str)
print("\nResults saved to outputs/analysis/baseline_comparison.json")
