"""
Poincaré ball geometry operations — pure numpy, numerically stable.

The Poincaré ball is the open unit ball {x in R^n : ||x|| < 1} with
the Riemannian metric g_x = (2/(1 - ||x||^2))^2 * I.
Curvature c = -1 (standard hyperbolic space).
"""

from __future__ import annotations

import numpy as np


class PoincareBall:
    """Hyperbolic geometry operations on the Poincaré ball model."""

    EPS = 1e-5
    MAX_NORM = 1.0 - 1e-5

    @staticmethod
    def distance(u: np.ndarray, v: np.ndarray) -> float:
        """Poincaré distance: d(u,v) = arcosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        diff_sq = float(np.sum((u - v) ** 2))
        u_sq = float(np.sum(u ** 2))
        v_sq = float(np.sum(v ** 2))
        denom = (1.0 - u_sq) * (1.0 - v_sq)
        if denom < 1e-12:
            denom = 1e-12
        arg = 1.0 + 2.0 * diff_sq / denom
        arg = max(arg, 1.0 + 1e-15)
        return float(np.arccosh(arg))

    @staticmethod
    def norm(v: np.ndarray) -> float:
        """Euclidean norm of a point on the ball."""
        return float(np.linalg.norm(v))

    @staticmethod
    def project(v: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Project point onto ball: clamp ||v|| < 1-eps."""
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        max_norm = 1.0 - eps
        if n >= max_norm:
            return v * (max_norm / n)
        return v

    @staticmethod
    def mobius_add(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Mobius addition: u ⊕ v."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        uv = float(np.dot(u, v))
        u_sq = float(np.sum(u ** 2))
        v_sq = float(np.sum(v ** 2))
        denom = 1.0 + 2.0 * uv + u_sq * v_sq
        if abs(denom) < 1e-12:
            denom = 1e-12
        num_u = (1.0 + 2.0 * uv + v_sq) * u
        num_v = (1.0 - u_sq) * v
        return PoincareBall.project(num_u + num_v) / denom

    @staticmethod
    def conformal_factor(v: np.ndarray) -> float:
        """Conformal factor lambda_v = 2 / (1 - ||v||^2)."""
        v = np.asarray(v, dtype=np.float64)
        v_sq = float(np.sum(v ** 2))
        return 2.0 / (1.0 - v_sq)

    @staticmethod
    def expmap(origin: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Exponential map from tangent space at origin to manifold."""
        origin = np.asarray(origin, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            return origin.copy()
        lam = PoincareBall.conformal_factor(origin)
        coeff = np.tanh(lam * v_norm / 2.0)
        direction = v / v_norm
        result = PoincareBall.mobius_add(origin, coeff * direction)
        return PoincareBall.project(result)

    @staticmethod
    def logmap(origin: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Logarithmic map from manifold to tangent space at origin."""
        origin = np.asarray(origin, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if np.allclose(origin, v, atol=1e-10):
            return np.zeros_like(v)
        minus_o = -origin
        diff = PoincareBall.mobius_add(minus_o, v)
        diff_norm = np.linalg.norm(diff)
        if diff_norm < 1e-10:
            return np.zeros_like(v)
        lam = PoincareBall.conformal_factor(origin)
        coeff = 2.0 * np.arctanh(min(diff_norm, 1.0 - 1e-10)) / (lam * diff_norm)
        return coeff * diff
