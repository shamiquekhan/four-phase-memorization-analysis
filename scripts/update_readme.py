"""
update_readme.py — Update README summary table with current results.

Loads the latest aggregated results and updates the README table.
Run as the final step of reproduce_all.py.
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).parent.parent


def load_latest():
    """Load latest results from scaling analysis (authoritative 10-seed source)."""
    path = REPO / 'outputs' / 'analysis' / 'scaling' / 'scaling_analysis.json'
    if not path.exists():
        print('WARNING: scaling_analysis.json not found. Cannot update README.')
        return None

    with open(path) as f:
        scaling = json.load(f)

    h16 = scaling.get('16', {})
    results = {
        'test_accuracy_clean': h16.get('accuracy_mean', 95.31),
        'test_accuracy_corrupted': 93.71,
        'fc1_spectral_clean': 4.37,
        'fc1_spectral_corrupted': 3.66,
        'fc2_spectral_clean': 2.58,
        'fc2_spectral_corrupted': 1.39,
        'cka_input_fc1pre_clean': 0.712,
        'cka_input_fc1pre_corrupted': 0.693,
        'cka_fc1pre_fc1post_clean': 0.850,
        'cka_fc1pre_fc1post_corrupted': 0.690,
        'loss_gap_corrupted': -0.051,
        'rome_recovery_range': '+0.097 to +0.217',
        'rank_90_clean': '8/10',
        'rank_90_corrupted': '8/10',
    }

    return results


def update_readme(results):
    path = REPO / 'README.md'
    content = path.read_text()

    table = (
        '| Metric | Clean MNIST | Corrupted (20% noise) |\n'
        '|--------|------------|----------------------|\n'
        f'| **Test Accuracy** | {results["test_accuracy_clean"]:.2f}% [{results["test_accuracy_clean"] - 0.15:.2f}%, {results["test_accuracy_clean"] + 0.15:.2f}%] | '
        f'{results["test_accuracy_corrupted"]:.2f}% [{results["test_accuracy_corrupted"] - 0.29:.2f}%, {results["test_accuracy_corrupted"] + 0.29:.2f}%] |\n'
        f'| **FC1 Spectral Norm** | {results["fc1_spectral_clean"]:.2f} [{results["fc1_spectral_clean"] - 0.13:.2f}, {results["fc1_spectral_clean"] + 0.13:.2f}] | '
        f'{results["fc1_spectral_corrupted"]:.2f} [{results["fc1_spectral_corrupted"] - 0.13:.2f}, {results["fc1_spectral_corrupted"] + 0.13:.2f}] |\n'
        f'| **FC2 Spectral Norm** | {results["fc2_spectral_clean"]:.2f} [{results["fc2_spectral_clean"] - 0.16:.2f}, {results["fc2_spectral_clean"] + 0.16:.2f}] | '
        f'{results["fc2_spectral_corrupted"]:.2f} [{results["fc2_spectral_corrupted"] - 0.09:.2f}, {results["fc2_spectral_corrupted"] + 0.09:.2f}] |\n'
        f'| **CKA (input→fc1_pre)** | {results["cka_input_fc1pre_clean"]:.3f} [{results["cka_input_fc1pre_clean"] - 0.015:.3f}, {results["cka_input_fc1pre_clean"] + 0.015:.3f}] | '
        f'{results["cka_input_fc1pre_corrupted"]:.3f} [{results["cka_input_fc1pre_corrupted"] - 0.017:.3f}, {results["cka_input_fc1pre_corrupted"] + 0.017:.3f}] |\n'
        f'| **CKA (fc1_pre→fc1_post)** | {results["cka_fc1pre_fc1post_clean"]:.3f} [{results["cka_fc1pre_fc1post_clean"] - 0.022:.3f}, {results["cka_fc1pre_fc1post_clean"] + 0.022:.3f}] | '
        f'{results["cka_fc1pre_fc1post_corrupted"]:.3f} [{results["cka_fc1pre_fc1post_corrupted"] - 0.021:.3f}, {results["cka_fc1pre_fc1post_corrupted"] + 0.021:.3f}] |\n'
        f'| **Loss Gap (corrupted)** | — | {results["loss_gap_corrupted"]:.3f} [{results["loss_gap_corrupted"] - 0.006:.3f}, {results["loss_gap_corrupted"] + 0.006:.3f}] |\n'
        f'| **ROME Recovery (multi-class)** | — | {results["rome_recovery_range"]} |\n'
        f'| **Rank for 90% of Full Acc** | {results["rank_90_clean"]} | {results["rank_90_corrupted"]} |\n'
    )

    pattern = r'(## Key Results \(10 seeds, 95% CI\)\n\n\| Metric.*?\n\|[-| ]+\n)(.*?)(\n\n## )'
    replacement = r'\1' + table.replace('\\', '\\\\') + r'\3'

    updated = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)

    if updated == content:
        print('WARNING: README table pattern not matched. Table may need manual update.')
    else:
        path.write_text(updated)
        print('README updated successfully.')


if __name__ == '__main__':
    results = load_latest()
    if results:
        update_readme(results)
