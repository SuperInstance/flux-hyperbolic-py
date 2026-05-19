"""Tests for Poincaré ball geometry operations."""

import numpy as np
import pytest

from flux_hyperbolic.geometry import PoincareBall


class TestDistance:
    def test_same_point_is_zero(self):
        p = np.array([0.1, 0.2, 0.3])
        assert PoincareBall.distance(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self):
        u = np.array([0.1, 0.2, 0.3])
        v = np.array([0.4, 0.5, 0.6])
        assert PoincareBall.distance(u, v) == pytest.approx(PoincareBall.distance(v, u))

    def test_positive_for_distinct(self):
        u = np.array([0.0, 0.0, 0.0])
        v = np.array([0.5, 0.0, 0.0])
        assert PoincareBall.distance(u, v) > 0.0

    def test_origin_distance_known(self):
        # d(0, v) = arcosh(1 + 2||v||^2 / (1-||v||^2))
        # For v=[0.5, 0, 0]: ||v||^2 = 0.25, arg = 1 + 2*0.25/0.75 = 1 + 0.6667 = 1.6667
        v = np.array([0.5, 0.0, 0.0])
        d = PoincareBall.distance(np.zeros(3), v)
        expected = np.arccosh(1.0 + 2.0 * 0.25 / 0.75)
        assert d == pytest.approx(expected, rel=1e-6)

    def test_boundary_points_far(self):
        u = np.array([0.999, 0.0])
        v = np.array([-0.999, 0.0])
        d = PoincareBall.distance(u, v)
        assert d > 5.0  # very far in hyperbolic space


class TestProject:
    def test_inside_ball_unchanged(self):
        v = np.array([0.1, 0.2, 0.3])
        projected = PoincareBall.project(v)
        np.testing.assert_array_almost_equal(projected, v)

    def test_outside_ball_clamped(self):
        v = np.array([2.0, 0.0, 0.0])
        projected = PoincareBall.project(v)
        assert np.linalg.norm(projected) < 1.0

    def test_unit_vector_clamped(self):
        v = np.array([1.0, 0.0, 0.0])
        projected = PoincareBall.project(v)
        assert np.linalg.norm(projected) < 1.0


class TestMobiusAdd:
    def test_left_identity(self):
        u = np.zeros(3)
        v = np.array([0.3, 0.1, -0.2])
        result = PoincareBall.mobius_add(u, v)
        np.testing.assert_array_almost_equal(result, v, decimal=5)

    def test_on_ball(self):
        u = np.array([0.3, 0.1, -0.2])
        v = np.array([0.1, 0.2, 0.1])
        result = PoincareBall.mobius_add(u, v)
        assert np.linalg.norm(result) < 1.0


class TestConformalFactor:
    def test_at_origin(self):
        assert PoincareBall.conformal_factor(np.zeros(3)) == pytest.approx(2.0)

    def test_grows_toward_boundary(self):
        cf_center = PoincareBall.conformal_factor(np.array([0.1, 0.0, 0.0]))
        cf_edge = PoincareBall.conformal_factor(np.array([0.9, 0.0, 0.0]))
        assert cf_edge > cf_center


class TestExpLogMap:
    def test_logmap_identity(self):
        origin = np.zeros(3)
        v = np.array([0.3, 0.1, -0.2])
        log_v = PoincareBall.logmap(origin, v)
        recovered = PoincareBall.expmap(origin, log_v)
        np.testing.assert_array_almost_equal(recovered, v, decimal=5)

    def test_expmap_identity(self):
        origin = np.zeros(3)
        v = np.array([0.3, 0.1, -0.2])
        exp_v = PoincareBall.expmap(origin, v)
        log_v = PoincareBall.logmap(origin, exp_v)
        np.testing.assert_array_almost_equal(log_v, v, decimal=5)

    def test_expmap_zero_vector(self):
        origin = np.array([0.1, 0.2, 0.3])
        result = PoincareBall.expmap(origin, np.zeros(3))
        np.testing.assert_array_almost_equal(result, origin, decimal=5)

    def test_logmap_same_point(self):
        p = np.array([0.3, 0.1, -0.2])
        result = PoincareBall.logmap(p, p)
        np.testing.assert_array_almost_equal(result, np.zeros(3), decimal=5)

    def test_roundtrip_nonzero_origin(self):
        origin = np.array([0.2, -0.1, 0.3])
        target = np.array([0.5, 0.1, -0.2])
        log_v = PoincareBall.logmap(origin, target)
        recovered = PoincareBall.expmap(origin, log_v)
        np.testing.assert_array_almost_equal(recovered, target, decimal=4)
