"""Tests for statistical metrics and model functionality."""
import pytest
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.model import MNISTNet
from src.utils.stats import compute_ci, aggregate_results


class TestComputeCI:
    """Tests for confidence interval computation."""

    def test_compute_ci_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, ci_low, ci_high = compute_ci(values)
        assert mean == pytest.approx(3.0)
        assert ci_low < mean < ci_high

    def test_compute_ci_single_value(self):
        values = [5.0]
        mean, ci_low, ci_high = compute_ci(values)
        assert mean == 5.0
        assert ci_low <= mean <= ci_high

    def test_compute_ci_empty(self):
        mean, ci_low, ci_high = compute_ci([])
        import math
        assert math.isnan(mean)

    def test_compute_ci_identical(self):
        values = [3.0, 3.0, 3.0]
        mean, ci_low, ci_high = compute_ci(values)
        assert mean == 3.0
        assert ci_low <= mean <= ci_high

    def test_compute_ci_two_values(self):
        values = [0.0, 10.0]
        mean, ci_low, ci_high = compute_ci(values)
        assert mean == 5.0
        assert ci_low <= mean <= ci_high

    def test_compute_ci_floats(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        mean, ci_low, ci_high = compute_ci(values)
        assert mean == pytest.approx(0.3)
        assert ci_low < mean < ci_high


class TestAggregateResults:
    """Tests for aggregate_results."""

    def test_aggregate_basic(self):
        results = {'accuracy': [95.0, 96.0, 97.0, 98.0, 99.0]}
        agg = aggregate_results(results)
        assert 'accuracy' in agg
        assert agg['accuracy']['mean'] == pytest.approx(97.0)
        assert agg['accuracy']['ci_lower'] < agg['accuracy']['mean'] < agg['accuracy']['ci_upper']

    def test_aggregate_single_value(self):
        results = {'loss': [0.5]}
        agg = aggregate_results(results)
        assert agg['loss']['mean'] == 0.5

    def test_aggregate_multi_metric(self):
        results = {
            'acc': [90.0, 92.0, 94.0],
            'loss': [0.5, 0.4, 0.3]
        }
        agg = aggregate_results(results)
        assert 'acc' in agg and 'loss' in agg
        assert 90.0 < agg['acc']['mean'] < 94.0
        assert 0.3 < agg['loss']['mean'] < 0.5


class TestMNISTNet:
    """Tests for MNISTNet model."""

    def test_model_creation(self):
        model = MNISTNet(784, 128, 10)
        assert model.input_dim == 784
        assert model.hidden_dim == 128
        assert model.output_dim == 10

    def test_forward_pass(self):
        model = MNISTNet(784, 128, 10)
        x = torch.randn(4, 784)
        out = model(x)
        assert out.shape == (4, 10)

    def test_forward_flattens(self):
        model = MNISTNet(784, 128, 10)
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)

    def test_forward_with_all_layers(self):
        model = MNISTNet(784, 128, 10)
        x = torch.randn(4, 784)
        result = model.forward_with_all_layers(x)
        assert set(result.keys()) == {'input', 'fc1_pre_activation', 'fc1_post_activation', 'output'}
        assert result['input'].shape == (4, 784)
        assert result['fc1_pre_activation'].shape == (4, 128)
        assert result['fc1_post_activation'].shape == (4, 128)
        assert result['output'].shape == (4, 10)

    def test_get_hidden(self):
        model = MNISTNet(784, 128, 10)
        x = torch.randn(4, 784)
        hidden = model.get_hidden(x)
        assert hidden.shape == (4, 128)

    def test_get_weight_info(self):
        model = MNISTNet(784, 128, 10)
        info = model.get_weight_info()
        assert 'fc1' in info
        assert 'fc2' in info
        assert info['fc1']['shape'] == (128, 784)
        assert info['fc2']['shape'] == (10, 128)
        expected_params = 784*128 + 128 + 128*10 + 10
        assert info['total_params'] == expected_params

    def test_save_and_load_checkpoint(self, tmp_path):
        model = MNISTNet(784, 128, 10)
        path = tmp_path / "model.pt"
        model.save_checkpoint(str(path), metadata={'test': 1})
        assert path.exists()

        loaded = MNISTNet.load_checkpoint(str(path))
        assert loaded.input_dim == 784
        assert loaded.hidden_dim == 128
        assert loaded.output_dim == 10
        assert loaded.activation_name == 'relu'
        assert loaded.training is False  # eval mode

    def test_count_parameters(self):
        model = MNISTNet(784, 128, 10)
        from src.models.model import count_parameters
        n = count_parameters(model)
        expected = 784*128 + 128 + 128*10 + 10
        assert n == expected

    def test_different_activations(self):
        relu_model = MNISTNet(784, 128, 10, activation='relu')
        tanh_model = MNISTNet(784, 128, 10, activation='tanh')
        gelu_model = MNISTNet(784, 128, 10, activation='gelu')
        x = torch.randn(4, 784)
        assert relu_model(x).shape == (4, 10)
        assert tanh_model(x).shape == (4, 10)
        assert gelu_model(x).shape == (4, 10)


class TestFDR:
    """Tests for Fisher Discriminant Ratio metric."""

    def test_fdr_perfect_separation(self):
        from src.utils.metrics import compute_fdr
        act = torch.tensor([[1.0, 0.0], [1.0, 0.1], [10.0, 0.0], [10.0, 0.1]])
        labels = torch.tensor([0, 0, 1, 1])
        fdr = compute_fdr(act, labels)
        assert fdr > 0

    def test_fdr_dimension_invariance(self):
        from src.utils.metrics import compute_fdr
        act = torch.randn(20, 4)
        labels = torch.tensor([0]*10 + [1]*10)
        fdr1 = compute_fdr(act, labels)
        act_scaled = act * 2.0
        fdr2 = compute_fdr(act_scaled, labels)
        assert abs(fdr1 - fdr2) < 1e-6

    def test_fdr_same_class(self):
        from src.utils.metrics import compute_fdr
        act = torch.randn(10, 4)
        labels = torch.zeros(10, dtype=torch.long)
        fdr = compute_fdr(act, labels)
        assert fdr == 0.0  # no between-class separation with single class

    def test_sigma_and_fdr_consistency(self):
        from src.utils.metrics import compute_sigma_and_fdr
        np.random.seed(42)
        act = np.random.randn(50, 8)
        labels = np.array([0]*25 + [1]*25)
        result = compute_sigma_and_fdr(act, labels)
        assert 'sigma' in result
        assert 'fdr' in result
        assert result['sigma'] > 0
        assert result['fdr'] > 0


class TestConfig:
    """Tests for configuration file."""

    def test_config_exists(self):
        config_path = Path(__file__).parent.parent / 'configs/experiment_config.yaml'
        assert config_path.exists()

    def test_config_valid_yaml(self):
        import yaml
        config_path = Path(__file__).parent.parent / 'configs/experiment_config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert 'model' in config
        assert 'corrupted_training' in config
        assert 'phase1' in config