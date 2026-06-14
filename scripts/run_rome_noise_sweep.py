"""
Noise rate sweep for ROME analysis + 0→8 multi-rank ablation.
Runs on all existing checkpoints and saves results.
"""
import sys, json, torch, numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from models.model import MNISTNet
from analysis.phase4_rome import compute_rome_for_all_classes
from analysis.rank_ablation import run_rome_rank_ablation
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.stats import compute_ci, paired_t_test

SEEDS = [42, 123, 456, 789, 1024]
NOISE_RATES = [0.1, 0.2, 0.4]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def load_model(ckpt_path):
    model = MNISTNet(784, 16, 10).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)['model_state_dict'])
    model.eval()
    return model

# ── Part 1: Noise rate sweep ──
print("=" * 60)
print("PART 1: NOISE RATE SWEEP ROME")
print("=" * 60)

sweep_results = {}
for noise in NOISE_RATES:
    print(f"\n--- Noise rate: {noise} ---")
    fc1_deltas = {c: [] for c in range(10)}
    fc2_deltas = {c: [] for c in range(10)}
    for seed in SEEDS:
        ckpt = Path(f'outputs/corrupted/noise_{noise}/seed_{seed}/final_model.pt')
        if not ckpt.exists():
            print(f"  Missing: seed {seed}")
            continue
        model = load_model(ckpt)
        fc1 = compute_rome_for_all_classes(model, train_loader, DEVICE, 'fc1')
        fc2 = compute_rome_for_all_classes(model, train_loader, DEVICE, 'fc2')
        for c in range(10):
            if c in fc1: fc1_deltas[c].append(fc1[c]['delta_norm'])
            if c in fc2: fc2_deltas[c].append(fc2[c]['delta_norm'])
        print(f"  Seed {seed}: fc1 avg={np.mean([fc1[c]['delta_norm'] for c in fc1]):.2f}, fc2 avg={np.mean([fc2[c]['delta_norm'] for c in fc2]):.2f}")

    print(f"\n  FC1 by class (noise={noise}):")
    for c in range(10):
        if fc1_deltas[c]:
            m, lo, hi = compute_ci(fc1_deltas[c])
            print(f"    Class {c}: {m:.2f} [{lo:.2f}, {hi:.2f}]")

    print(f"  FC2 by class (noise={noise}):")
    for c in range(10):
        if fc2_deltas[c]:
            m, lo, hi = compute_ci(fc2_deltas[c])
            print(f"    Class {c}: {m:.2f} [{lo:.2f}, {hi:.2f}]")

    sweep_results[str(noise)] = {
        'fc1': {str(c): fc1_deltas[c] for c in range(10) if fc1_deltas[c]},
        'fc2': {str(c): fc2_deltas[c] for c in range(10) if fc2_deltas[c]},
    }

json.dump(sweep_results, open('outputs/analysis/noise_sweep_rome.json', 'w'), indent=2, default=str)
print("\nNoise sweep saved to outputs/analysis/noise_sweep_rome.json")

# ── Part 2: 0→8 multi-rank ROME ablation ──
print("\n" + "=" * 60)
print("PART 2: 0→8 MULTI-RANK ROME ABLATION")
print("=" * 60)

ablation_results = {}
for seed in SEEDS:
    ckpt = Path(f'outputs/targeted_corrupted/src0_tgt8/seed_{seed}/final_model.pt')
    if not ckpt.exists():
        print(f"  Missing checkpoint for seed {seed}")
        continue
    model = load_model(ckpt)
    result = run_rome_rank_ablation(model, test_loader, DEVICE, source_class=0, target_class=8)
    ablation_results[str(seed)] = result
    print(f"  Seed {seed}: ranks={result['ranks']}")
    for k in result['per_class_acc']:
        src_acc = result['per_class_acc'][k].get('0', 0)
        print(f"    Rank {k}: source class 0 acc={src_acc:.4f}")

json.dump(ablation_results, open('outputs/analysis/rome_08_rank_ablation.json', 'w'), indent=2, default=str)
print("\n0→8 rank ablation saved to outputs/analysis/rome_08_rank_ablation.json")

# ── Summary ──
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Clean model ROME for baseline comparison
print("\nClean model ROME (baseline, fc2):")
clean_fc2 = {c: [] for c in range(10)}
for seed in SEEDS:
    ckpt = Path(f'outputs/clean/seed_{seed}/final_model.pt')
    if not ckpt.exists(): continue
    model = load_model(ckpt)
    fc2 = compute_rome_for_all_classes(model, train_loader, DEVICE, 'fc2')
    for c in range(10):
        if c in fc2: clean_fc2[c].append(fc2[c]['delta_norm'])
for c in range(10):
    if clean_fc2[c]:
        m, lo, hi = compute_ci(clean_fc2[c])
        print(f"  Class {c}: {m:.2f} [{lo:.2f}, {hi:.2f}]")

# Compare clean vs each noise rate
print("\nClean vs noise rate comparison (fc2 avg):")
for noise in NOISE_RATES:
    clean_avgs = [np.mean([clean_fc2[c][i] for c in range(10) if clean_fc2[c]]) for i in range(len(SEEDS))]
    corrupt_avgs = []
    for i, seed in enumerate(SEEDS):
        vals = sweep_results[str(noise)]['fc2']
        if vals:
            class_vals = [vals[str(c)][i] for c in range(10) if str(c) in vals and len(vals[str(c)]) > i]
            if class_vals:
                corrupt_avgs.append(np.mean(class_vals))
    if len(clean_avgs) > 1 and len(corrupt_avgs) > 1:
        clean_mean = np.mean(clean_avgs)
        corrupt_mean = np.mean(corrupt_avgs)
        ratio = clean_mean / corrupt_mean if corrupt_mean > 0 else float('inf')
        t, p = paired_t_test(clean_avgs[:len(corrupt_avgs)], corrupt_avgs)
        print(f"  Noise={noise}: clean={clean_mean:.2f} corrupt={corrupt_mean:.2f} ratio={ratio:.2f}x p={p:.6f}")
