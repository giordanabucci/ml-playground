import pytest
import numpy as np
from app.domain.circle import CircleDataset
from app.domain.spiral import SpiralDataset
from app.domain.gaussian import GaussianDataset
from app.domain.xor import XORDataset
from app.domain.linear import LinearDataset
from app.domain.diagonal import DiagonalDataset

class TestCircleDataset:
    @pytest.mark.parametrize("n_samples, noise", [
        (100, 0.0),
        (500, 0.1),
        (1000, 1.0)
    ])
    def test_circle_valid_parameters(self, n_samples, noise):
        dataset = CircleDataset(n_samples=n_samples, noise=noise)
        assert dataset.X.shape == (n_samples, 2)
        assert dataset.y.shape == (n_samples,)
        assert set(np.unique(dataset.y)).issubset({-1, 1})
    
    def test_circle_invalid_samples(self):
        with pytest.raises(ValueError):
            CircleDataset(n_samples=-10, noise=0.1)
    

class TestSpiralDataset:
    @pytest.mark.parametrize("n_samples, noise", [
        (100, 0.0),
        (500, 0.1),
        (1000, 1.0)
    ])
    def test_spiral_valid_parameters(self, n_samples, noise):
        dataset = SpiralDataset(n_samples=n_samples, noise=noise)
        assert dataset.X.shape == (n_samples, 2)
        assert dataset.y.shape == (n_samples,)
        assert set(np.unique(dataset.y)).issubset({-1, 1})

    def test_spiral_invalid_samples(self):
        with pytest.raises(ValueError):
            SpiralDataset(n_samples=-5, noise=0.1)

class TestGaussianDataset:
    @pytest.mark.parametrize("n_samples, noise", [
        (100, 0.0),
        (500, 0.1),
        (1000, 1.0)
    ])
    def test_gaussian_valid_parameters(self, n_samples, noise):
        dataset = GaussianDataset(n_samples=n_samples, noise=noise)
        assert dataset.X.shape == (n_samples, 2)
        assert dataset.y.shape == (n_samples,)
        assert set(np.unique(dataset.y)).issubset({-1, 1})

    def test_gaussian_invalid_samples(self):
        with pytest.raises(ValueError):
            GaussianDataset(n_samples=-1, noise=0.1)

class TestXORDataset:
    @pytest.mark.parametrize("n_samples, noise", [
        (100, 0.0),
        (500, 0.1),
        (1000, 1.0)
    ])
    def test_xor_valid_parameters(self, n_samples, noise):
        dataset = XORDataset(n_samples=n_samples, noise=noise)
        assert dataset.X.shape == (n_samples, 2)
        assert dataset.y.shape == (n_samples,)
        assert set(np.unique(dataset.y)).issubset({-1, 1})

    def test_xor_invalid_samples(self):
        with pytest.raises(ValueError):
            XORDataset(n_samples=-100, noise=0.1)

class TestLinearDataset:
    @pytest.mark.parametrize("n_samples, noise", [
        (100, 0.0),
        (500, 0.1),
        (1000, 1.0)
    ])
    def test_linear_valid_parameters(self, n_samples, noise):
        dataset = LinearDataset(n_samples=n_samples, noise=noise)
        assert dataset.X.shape == (n_samples, 2)
        assert dataset.y.shape == (n_samples,)
        assert set(np.unique(dataset.y)).issubset({-1, 1})

    def test_linear_invalid_samples(self):
        with pytest.raises(ValueError):
            LinearDataset(n_samples=-50, noise=0.1)

class TestDiagonalDataset:
    @pytest.mark.parametrize("n_samples, noise", [
        (100, 0.0),
        (500, 0.1),
        (1000, 1.0)
    ])
    def test_diagonal_valid_parameters(self, n_samples, noise):
        dataset = DiagonalDataset(n_samples=n_samples, noise=noise)
        assert dataset.X.shape == (n_samples, 2)
        assert dataset.y.shape == (n_samples,)
        assert set(np.unique(dataset.y)).issubset({-1, 1})

    def test_diagonal_invalid_samples(self):
        with pytest.raises(ValueError):
            DiagonalDataset(n_samples=-1, noise=0.1)