"""Tests for flux_hyperbolic.consensus — FrechetMean."""

import pytest
import numpy as np
from flux_hyperbolic.consensus import FrechetMean


class TestFrechetMean:
    def test_single_point(self):
        p = np.array([0.1, 0.2, 0.05])
        result = FrechetMean.compute([p])
        assert result.shape == (3,)
        # Should be close to the input point
        assert np.linalg.norm(result - p) < 0.1

    def test_multiple_points(self):
        points = [np.array([0.1, 0.1]), np.array([0.2, 0.1]), np.array([0.1, 0.2])]
        result = FrechetMean.compute(points)
        assert result.shape == (2,)
        # Should be inside the Poincare ball
        assert np.linalg.norm(result) < 1.0

    def test_with_weights(self):
        p1 = np.array([0.1, 0.0])
        p2 = np.array([0.5, 0.0])
        # Heavy weight on p1 → result should be closer to p1
        result = FrechetMean.compute([p1, p2], weights=np.array([0.9, 0.1]))
        assert np.linalg.norm(result - p1) < np.linalg.norm(result - p2)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            FrechetMean.compute([])

    def test_convergence(self):
        # Two nearby points should converge
        p1 = np.array([0.05, 0.05])
        p2 = np.array([0.06, 0.06])
        result = FrechetMean.compute([p1, p2])
        assert np.linalg.norm(result) < 1.0

    def test_symmetric(self):
        p1 = np.array([0.1, 0.2])
        p2 = np.array([0.2, 0.1])
        r1 = FrechetMean.compute([p1, p2])
        r2 = FrechetMean.compute([p2, p1])
        np.testing.assert_allclose(r1, r2, atol=1e-5)
