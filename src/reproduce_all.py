#!/usr/bin/env python3
"""
Single script to reproduce the full memorization analysis pipeline.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import time


def run(cmd, desc, cwd=None):
    print(f"\n{'='*60}")
    print(f"Step: {desc}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, cwd=cwd or '.')
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"Completed in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description='Reproduce full memorization analysis pipeline')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds')
    parser.add_argument('--skip-training', action='store_true', help='Skip training (use existing checkpoints)')
    parser.add_argument('--skip-scaling', action='store_true', help='Skip scaling experiments')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip analysis')
    parser.add_argument('--skip-figures', action='store_true', help='Skip figure generation')
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parent.parent
    seeds = list(range(args.seeds))
    
    # Step 1: Train on clean MNIST
    if not args.skip_training:
        for seed in seeds:
            run(
                f"python src/training/train_clean.py --config {args.config} --seed {seed} --output-dir {args.output_dir}/clean",
                f"Training clean model (seed {seed})",
                cwd=root
            )
        
        # Step 2: Train on corrupted MNIST
        for seed in seeds:
            run(
                f"python src/training/train_corrupted.py --config {args.config} --seed {seed} --noise-rate 0.2 --output-dir {args.output_dir}/corrupted",
                f"Training corrupted model (seed {seed})",
                cwd=root
            )
        
        # Step 3: Train targeted corrupted models (for ROME)
        for seed in seeds[:5]:
            for cfg_src, cfg_tgt in [(7, 1), (1, 7), (5, 6), (0, 8)]:
                run(
                    f"python src/training/train_targeted_corrupted.py --config {args.config} --seed {seed} --source {cfg_src} --target {cfg_tgt} --output-dir {args.output_dir}/targeted_corrupted",
                    f"Training targeted corrupted s{cfg_src}→t{cfg_tgt} (seed {seed})",
                    cwd=root
                )

        # Step 4: Scaling experiment
        if not args.skip_scaling:
            run(
                f"python src/scaling/train_scaling.py --config {args.config} --hidden-dims 16 32 64 128 256 --seeds {' '.join(map(str, seeds))} --epochs 20 --output-dir {args.output_dir}/scaling",
                "Running scaling experiments",
                cwd=root
            )
    
    # Step 5: Phase 1 - Basic analysis
    if not args.skip_analysis:
        for tag, checkpoint_dir in [('clean', f'{args.output_dir}/clean'), ('corrupted', f'{args.output_dir}/corrupted/noise_0.2')]:
            run(
                f"python src/analysis/phase1_basic.py --config {args.config} --checkpoint-dir {checkpoint_dir} --seeds {' '.join(map(str, seeds))} --output-dir {args.output_dir}/analysis/phase1_{tag}",
                f"Phase 1 analysis ({tag})",
                cwd=root
            )
        
        # Step 6: Phase 2 - Representation analysis
        for tag, checkpoint_dir in [('clean', f'{args.output_dir}/clean'), ('corrupted', f'{args.output_dir}/corrupted/noise_0.2')]:
            run(
                f"python src/analysis/phase2_representation.py --config {args.config} --checkpoint-dir {checkpoint_dir} --seeds {' '.join(map(str, seeds))} --output-dir {args.output_dir}/analysis/phase2_{tag}",
                f"Phase 2 analysis ({tag})",
                cwd=root
            )
        
        # Step 7: Phase 3 - Influence functions
        for tag, checkpoint_dir in [('clean', f'{args.output_dir}/clean'), ('corrupted', f'{args.output_dir}/corrupted/noise_0.2')]:
            run(
                f"python src/analysis/phase3_influence.py --config {args.config} --checkpoint-dir {checkpoint_dir} --seeds {' '.join(map(str, seeds[:5]))} --output-dir {args.output_dir}/analysis/phase3_{tag}",
                f"Phase 3 analysis ({tag})",
                cwd=root
            )
        
        # Step 8: Phase 4 - ROME analysis
        for tag, checkpoint_dir in [('clean', f'{args.output_dir}/clean'), ('corrupted', f'{args.output_dir}/corrupted/noise_0.2')]:
            run(
                f"python src/analysis/phase4_rome.py --config {args.config} --checkpoint-dir {checkpoint_dir} --seeds {' '.join(map(str, seeds))} --output-dir {args.output_dir}/analysis/phase4_{tag}",
                f"Phase 4 analysis ({tag})",
                cwd=root
            )
        
        # Step 8: Multi-class ROME validation (targeted corruption)
        run(
            f"python src/analysis/multiclass_rome.py --config {args.config} --checkpoint-dir {args.output_dir}/targeted_corrupted --seeds {' '.join(map(str, seeds[:5]))} --output-dir {args.output_dir}/analysis/multiclass_rome",
            "Multi-class ROME validation (targeted corruption)",
            cwd=root
        )
        
        # Step 10: Rank ablation
        for tag, checkpoint_dir in [('clean', f'{args.output_dir}/clean'), ('corrupted', f'{args.output_dir}/corrupted/noise_0.2')]:
            run(
                f"python src/analysis/rank_ablation.py --config {args.config} --checkpoint-dir {checkpoint_dir} --seeds {' '.join(map(str, seeds[:5]))} --output-dir {args.output_dir}/analysis/rank_ablation_{tag}",
                f"Rank ablation study ({tag})",
                cwd=root
            )
        
        # Step 11: Scaling analysis
        if not args.skip_scaling:
            run(
                f"python src/scaling/analyze_scaling.py --results-dir {args.output_dir}/scaling --output-dir {args.output_dir}/analysis/scaling",
                "Scaling analysis",
                cwd=root
            )
    
    # Step 12: Generate figures
    if not args.skip_figures:
        run(
            f"python src/analysis/visualizations.py --results-dir {args.output_dir} --output-dir {args.output_dir}/figures --config {args.config}",
            "Generating figures",
            cwd=root
        )
    
    # Step 13: Run tests
    run(
        "python -m pytest tests/ -v --tb=short",
        "Running unit tests",
        cwd=root
    )
    
    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")
    print(f"All outputs in: {args.output_dir}")
    print(f"Run `python -m pytest tests/ -v` to verify results.")


if __name__ == '__main__':
    main()