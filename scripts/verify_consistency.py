"""
verify_consistency.py — Verify README, RESULTS, and PAPER values match.

Checks across all documentation sources that reported numbers agree
within tolerance. Run after pipeline changes to catch stale values.
"""

import json
import re
import sys
from pathlib import Path

TOLERANCE = 0.005  # 0.5 percentage point

REPO = Path(__file__).parent.parent


def load_scaling():
    path = REPO / 'outputs' / 'analysis' / 'scaling' / 'scaling_analysis.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_readme_table():
    path = REPO / 'README.md'
    text = path.read_text()

    # Find the key results table
    lines = text.split('\n')
    in_table = False
    table_data = {}
    for i, line in enumerate(lines):
        if '| **Test Accuracy**' in line:
            parts = [p.strip() for p in line.split('|')]
            # parts: ['', '**Test Accuracy**', '95.31% [95.16%, 95.46%]', '93.71% [93.42%, 94.00%]', '']
            clean_str = parts[2].split('%')[0] if '%' in parts[2] else parts[2].split('[')[0]
            corr_str = parts[3].split('%')[0] if '%' in parts[3] else parts[3].split('[')[0]
            table_data['test_accuracy_clean'] = float(clean_str)
            table_data['test_accuracy_corrupted'] = float(corr_str)
        elif '| **CKA (input→fc1_pre)**' in line:
            parts = [p.strip() for p in line.split('|')]
            clean_str = parts[2].split('[')[0]
            corr_str = parts[3].split('[')[0]
            table_data['cka_input_fc1pre_clean'] = float(clean_str)
            table_data['cka_input_fc1pre_corrupted'] = float(corr_str)
        elif '| **CKA (fc1_pre→fc1_post)**' in line:
            parts = [p.strip() for p in line.split('|')]
            clean_str = parts[2].split('[')[0]
            corr_str = parts[3].split('[')[0]
            table_data['cka_fc1pre_fc1post_clean'] = float(clean_str)
            table_data['cka_fc1pre_fc1post_corrupted'] = float(corr_str)
        elif '| **Rank for 90% of Full Acc**' in line:
            parts = [p.strip() for p in line.split('|')]
            table_data['rank_90_clean'] = parts[2]
            table_data['rank_90_corrupted'] = parts[3]
    return table_data


def checks():
    passed, failed = 0, 0
    results = []

    scaling = load_scaling()
    readme = extract_readme_table()

    # Check README vs scaling h=16 accuracy
    if scaling and readme:
        h16_acc = scaling['16']['accuracy_mean']
        readme_acc = readme.get('test_accuracy_clean')
        if readme_acc is not None:
            if abs(h16_acc - readme_acc) <= TOLERANCE * 100:
                results.append(f'  PASS: README clean acc {readme_acc} ≈ scaling h=16 {h16_acc:.2f}')
                passed += 1
            else:
                results.append(f'  FAIL: README clean acc {readme_acc} ≠ scaling h=16 {h16_acc:.2f}')
                failed += 1

    # Check RESULTS.md vs scaling
    results_path = REPO / 'RESULTS.md'
    if results_path.exists() and scaling:
        text = results_path.read_text()
        for h_str in ['16', '32', '64', '128', '256', '512', '1024']:
            h = int(h_str)
            if h_str not in scaling:
                continue
            s = scaling[h_str]
            # Find table row for this hidden dim (4 numeric columns: Mono, Circuit, Sparsity, Acc)
            pattern = rf'\|\s*{h}\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+%?)'
            match = re.search(pattern, text)
            if match:
                results_mono = float(match.group(1))
                results_circuit = float(match.group(2))
                results_sparsity = float(match.group(3))
                passed_local = 0
                failed_local = 0
                if abs(results_mono - s['mono_fraction_mean']) <= TOLERANCE:
                    results.append(f'  PASS: RESULTS.md h={h} mono {results_mono} ≈ scaling {s["mono_fraction_mean"]:.3f}')
                    passed_local += 1
                else:
                    results.append(f'  FAIL: RESULTS.md h={h} mono {results_mono} ≠ scaling {s["mono_fraction_mean"]:.3f}')
                    failed_local += 1
                if abs(results_circuit - s['circuit_size_mean']) <= TOLERANCE * 10:  # circuit displayed with 1dp
                    results.append(f'  PASS: RESULTS.md h={h} circuit {results_circuit} ≈ scaling {s["circuit_size_mean"]:.2f}')
                    passed_local += 1
                else:
                    results.append(f'  FAIL: RESULTS.md h={h} circuit {results_circuit} ≠ scaling {s["circuit_size_mean"]:.2f}')
                    failed_local += 1
                if abs(results_sparsity - s['sparsity_mean']) <= TOLERANCE:
                    results.append(f'  PASS: RESULTS.md h={h} sparsity {results_sparsity} ≈ scaling {s["sparsity_mean"]:.3f}')
                    passed_local += 1
                else:
                    results.append(f'  FAIL: RESULTS.md h={h} sparsity {results_sparsity} ≠ scaling {s["sparsity_mean"]:.3f}')
                    failed_local += 1
                passed += passed_local
                failed += failed_local

    # Check PAPER.md vs scaling h=16
    paper_path = REPO / 'PAPER.md'
    if paper_path.exists() and scaling:
        text = paper_path.read_text()
        s16 = scaling['16']
        # Check accuracy
        acc_match = re.search(r'Clean test accuracy\s*[|].*?([\d.]+)%', text)
        if acc_match:
            paper_acc = float(acc_match.group(1))
            if abs(paper_acc - s16['accuracy_mean']) <= TOLERANCE * 100:
                results.append(f'  PASS: PAPER.md clean acc {paper_acc}% ≈ scaling {s16["accuracy_mean"]:.2f}%')
                passed += 1
            else:
                results.append(f'  FAIL: PAPER.md clean acc {paper_acc}% ≠ scaling {s16["accuracy_mean"]:.2f}%')
                failed += 1

    print('=== CONSISTENCY VERIFICATION ===\n')
    for r in results:
        print(r)
    print(f'\n=== SUMMARY: {passed} passed, {failed} failed ===')
    return failed == 0


if __name__ == '__main__':
    sys.exit(0 if checks() else 1)
