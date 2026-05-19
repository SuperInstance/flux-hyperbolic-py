"""Tests for capability routing and Frechet mean."""

import numpy as np
import pytest

from flux_hyperbolic.capability import CapabilitySpace, TaskRouter
from flux_hyperbolic.consensus import FrechetMean
from flux_hyperbolic.geometry import PoincareBall


class TestCapabilitySpace:
    def test_add_model(self):
        space = CapabilitySpace()
        mp = space.add_model("test", np.ones(8) * 0.1)
        assert mp.name == "test"
        assert "test" in space.models

    def test_add_model_wrong_dim(self):
        space = CapabilitySpace()
        with pytest.raises(AssertionError):
            space.add_model("bad", np.ones(5))

    def test_add_general_model(self):
        space = CapabilitySpace()
        mp = space.add_general_model("gen", np.random.randn(8), norm=0.1)
        assert mp.norm < 0.2

    def test_add_specialist_model(self):
        space = CapabilitySpace()
        mp = space.add_specialist_model("spec", np.random.randn(8), norm=0.85)
        assert mp.norm > 0.7

    def test_nearest_model(self):
        space = CapabilitySpace()
        space.add_model("a", np.array([0.5, 0, 0, 0, 0, 0, 0, 0]))
        space.add_model("b", np.array([0, 0, 0.5, 0, 0, 0, 0, 0]))
        nearest = space.nearest_model(np.array([0.4, 0, 0, 0, 0, 0, 0, 0]))
        assert nearest[0][0] == "a"

    def test_distance(self):
        space = CapabilitySpace()
        space.add_model("a", np.zeros(8))
        space.add_model("b", np.ones(8) * 0.5)
        d = space.distance("a", "b")
        assert d > 0.0


class TestTaskRouter:
    def _make_space(self):
        space = CapabilitySpace()
        rng = np.random.RandomState(42)
        space.add_general_model("gen-1", rng.randn(8), norm=0.1)
        space.add_general_model("gen-2", rng.randn(8), norm=0.15)
        space.add_specialist_model("spec-code", rng.randn(8), norm=0.85)
        space.add_specialist_model("spec-math", rng.randn(8), norm=0.90)
        return space

    def test_route_task_returns_result(self):
        space = self._make_space()
        router = TaskRouter(space)
        result = router.route_task(0, np.random.randn(8), 0.5)
        assert result.hyperbolic_model in space.models
        assert result.euclidean_model in space.models

    def test_route_batch(self):
        space = self._make_space()
        router = TaskRouter(space)
        tasks = [(i, np.random.randn(8), 0.5) for i in range(10)]
        results = router.route_batch(tasks)
        assert len(results) == 10
        assert all(isinstance(r.hyperbolic_distance, float) for r in results)

    def test_embedding_on_ball(self):
        space = self._make_space()
        router = TaskRouter(space)
        emb = router.embed_task(np.random.randn(8), 0.8)
        assert np.linalg.norm(emb) < 1.0


class TestFrechetMean:
    def test_single_point(self):
        p = np.array([0.3, 0.1, -0.2, 0.0, 0.1, 0.2, -0.1, 0.0])
        mean = FrechetMean.compute([p])
        np.testing.assert_array_almost_equal(mean, p, decimal=5)

    def test_two_points_midpoint(self):
        p1 = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        p2 = np.array([-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mean = FrechetMean.compute([p1, p2])
        assert np.linalg.norm(mean) < np.linalg.norm(p1)

    def test_on_ball(self):
        rng = np.random.RandomState(123)
        points = [PoincareBall.project(rng.randn(8) * 0.5) for _ in range(10)]
        mean = FrechetMean.compute(points)
        assert np.linalg.norm(mean) < 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            FrechetMean.compute([])

    def test_weighted_mean(self):
        p1 = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        p2 = np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Weight p1 much more -> mean closer to p1
        mean_heavy_p1 = FrechetMean.compute([p1, p2], weights=np.array([0.99, 0.01]))
        mean_equal = FrechetMean.compute([p1, p2], weights=np.array([0.5, 0.5]))
        assert mean_heavy_p1[0] > mean_equal[0]  # closer to p1 in first coord
