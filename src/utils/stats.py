"""Statistical utilities: CI computation and multi-seed experiment runner."""

import numpy as np
import torch
from scipy import stats
from typing import Callable, Dict, List, Any, Tuple


SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 5555, 7777, 9999]


def compute_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval for a list of measurements across seeds.

    Args:
        values: List of scalar measurements (one per seed)
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    n = len(values)
    if n < 2:
        return float(np.mean(values)), float(np.mean(values)), float(np.mean(values))

    mean = np.mean(values)
    se = stats.sem(values)
    ci = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return float(mean), float(mean - ci), float(mean + ci)


def run_with_seeds(experiment_fn: Callable, seeds: List[int] = None, **kwargs) -> Dict[str, List[float]]:
    """
    Run any experiment function across multiple seeds.

    Args:
        experiment_fn: Callable that takes seed as first arg and returns dict of metrics
        seeds: List of integer seeds (defaults to SEEDS)
        **kwargs: Additional arguments passed to experiment_fn

    Returns:
        Dict of metric_name -> list of values across seeds
    """
    if seeds is None:
        seeds = SEEDS

    results: Dict[str, List[float]] = {}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        metrics = experiment_fn(seed=seed, **kwargs)
        for k, v in metrics.items():
            results.setdefault(k, []).append(float(v))
    return results


def aggregate_results(raw_results: Dict[str, List[float]], confidence: float = 0.95) -> Dict[str, Dict[str, float]]:
    """
    Aggregate raw multi-seed results into mean ± CI format.

    Args:
        raw_results: Dict of metric_name -> list of values across seeds
        confidence: Confidence level for CI

    Returns:
        Dict of metric_name -> {'mean': float, 'ci_lower': float, 'ci_upper': float, 'std': float}
    """
    aggregated = {}
    for metric, values in raw_results.items():
        mean, ci_lower, ci_upper = compute_ci(values, confidence)
        aggregated[metric] = {
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': float(np.std(values)),
            'values': values
        }
    return aggregated


def format_mean_ci(aggregated: Dict[str, Dict[str, float]], metric: str, precision: int = 3) -> str:
    """Format a metric as 'mean ± ci' string."""
    m = aggregated[metric]
    return f"{m['mean']:.{precision}f} ± {m['ci_upper'] - m['mean']:.{precision}f}"


def paired_t_test(values_a: List[float], values_b: List[float]) -> Tuple[float, float]:
    """
    Perform paired t-test between two sets of measurements.

    Returns:
        t-statistic, p-value
    """
    t_stat, p_val = stats.ttest_rel(values_a, values_b)
    return float(t_stat), float(p_val)


def one_sample_t_test(values: List[float], popmean: float = 0.0) -> Tuple[float, float]:
    """
    Perform one-sample t-test against a population mean.

    Returns:
        t-statistic, p-value
    """
    t_stat, p_val = stats.ttest_1samp(values, popmean)
    return float(t_stat), float(p_val)