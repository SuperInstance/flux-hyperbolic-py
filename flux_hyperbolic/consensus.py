"""
FrechetMean — hyperbolic centroid for fleet consensus.

Iterative tangent-space averaging on the Poincaré ball:
  1. Pick initial estimate
  2. Logmap all points -> tangent space at estimate
  3. Weighted Euclidean mean in tangent space
  4. Expmap back -> new estimate
  5. Repeat until convergence
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from flux_hyperbolic.geometry import PoincareBall


class FrechetMean:
    """Compute the Frechet mean (centroid) in hyperbolic space."""

    MAX_ITER = 50
    TOL = 1e-7

    @staticmethod
    def compute(points: List[np.ndarray],
                weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the weighted Frechet mean of points on the ball."""
        if not points:
            raise ValueError("Need at least one point")
        points = [PoincareBall.project(np.asarray(p, dtype=np.float64)) for p in points]

        if weights is None:
            weights = np.ones(len(points)) / len(points)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights / weights.sum()

        estimate = sum(w * p for w, p in zip(weights, points))
        estimate = PoincareBall.project(estimate)

        for _ in range(FrechetMean.MAX_ITER):
            tangent = [PoincareBall.logmap(estimate, p) for p in points]
            tangent_mean = sum(w * t for w, t in zip(weights, tangent))
            if np.linalg.norm(tangent_mean) < FrechetMean.TOL:
                break
            estimate = PoincareBall.expmap(estimate, tangent_mean)
            estimate = PoincareBall.project(estimate)

        return estimate
