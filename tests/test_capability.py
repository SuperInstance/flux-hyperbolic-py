"""Tests for flux_hyperbolic.capability — CapabilitySpace, ModelPoint, TaskRouter."""

import pytest
import numpy as np
from flux_hyperbolic.capability import (
    CapabilitySpace, ModelPoint, TaskRouter, RoutingResult,
)


class TestModelPoint:
    def test_specialization_general(self):
        mp = ModelPoint(name="gpt", coords=np.zeros(8) * 0.1)
        assert mp.specialization_level in ("general", "moderate", "specialist")

    def test_conformal_factor(self):
        mp = ModelPoint(name="test", coords=np.array([0.1] * 8))
        cf = mp.conformal_factor
        assert cf > 0


class TestCapabilitySpace:
    def test_add_model(self):
        space = CapabilitySpace()
        mp = space.add_model("test", np.array([0.1] * 8))
        assert mp.name == "test"

    def test_add_model_wrong_dim(self):
        space = CapabilitySpace()
        with pytest.raises(AssertionError):
            space.add_model("test", np.array([0.1] * 5))

    def test_add_general_model(self):
        space = CapabilitySpace()
        mp = space.add_general_model("generalist", np.array([1.0] * 8))
        assert mp.norm < 0.3

    def test_add_specialist_model(self):
        space = CapabilitySpace()
        mp = space.add_specialist_model("specialist", np.array([1.0] * 8))
        assert mp.norm >= 0.7

    def test_distance(self):
        space = CapabilitySpace()
        space.add_model("a", np.array([0.1] * 8))
        space.add_model("b", np.array([0.5] * 8))
        d = space.distance("a", "b")
        assert d > 0

    def test_nearest_model(self):
        space = CapabilitySpace()
        space.add_model("a", np.array([0.1] * 8))
        space.add_model("b", np.array([0.8] * 8))
        # Query near a should return a
        results = space.nearest_model(np.array([0.1] * 8), n=1)
        assert results[0][0] == "a"


class TestTaskRouter:
    def _make_space(self):
        space = CapabilitySpace()
        space.add_model("general", np.array([0.1] * 8))
        space.add_specialist_model("math_spec", np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float))
        return space

    def test_embed_task(self):
        space = self._make_space()
        router = TaskRouter(space)
        emb = router.embed_task(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float))
        assert np.linalg.norm(emb) < 1.0

    def test_route_task(self):
        space = self._make_space()
        router = TaskRouter(space)
        result = router.route_task(1, np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float))
        assert isinstance(result, RoutingResult)
        assert result.task_id == 1
        assert isinstance(result.agree, bool)

    def test_route_batch(self):
        space = self._make_space()
        router = TaskRouter(space)
        tasks = [(1, np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float), 0.5),
                 (2, np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=float), 0.5)]
        results = router.route_batch(tasks)
        assert len(results) == 2
