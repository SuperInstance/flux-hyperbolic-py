"""
CapabilitySpace and TaskRouter — 8D Poincaré ball for model capability routing.

Dimensions (constraint-space axes):
  0: reasoning_depth   1: code_generation   2: math_precision
  3: context_window    4: speed              5: creativity
  6: safety            7: multilingual
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from flux_hyperbolic.geometry import PoincareBall


@dataclass
class ModelPoint:
    """A model embedded as a point in the capability space.

    Norm encodes specialization level:
      ||v|| < 0.2 -> generalist
      0.2 <= ||v|| < 0.7 -> moderate specialist
      ||v|| >= 0.7 -> high specialist
    """
    name: str
    coords: np.ndarray
    specialization_label: str = ""

    @property
    def norm(self) -> float:
        return PoincareBall.norm(self.coords)

    @property
    def specialization_level(self) -> str:
        n = self.norm
        if n < 0.2:
            return "general"
        elif n < 0.7:
            return "moderate"
        else:
            return "specialist"

    @property
    def conformal_factor(self) -> float:
        return PoincareBall.conformal_factor(self.coords)


class CapabilitySpace:
    """8-dimensional Poincaré ball for model capability routing."""

    DIM = 8

    def __init__(self) -> None:
        self.models: Dict[str, ModelPoint] = {}

    def add_model(self, name: str, coords: np.ndarray,
                  specialization_label: str = "") -> ModelPoint:
        coords = np.asarray(coords, dtype=np.float64)
        assert coords.shape == (self.DIM,), f"Expected {self.DIM}D, got {coords.shape}"
        coords = PoincareBall.project(coords)
        mp = ModelPoint(name=name, coords=coords,
                        specialization_label=specialization_label)
        self.models[name] = mp
        return mp

    def add_general_model(self, name: str, direction: np.ndarray,
                          norm: float = 0.1) -> ModelPoint:
        d = np.asarray(direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-10)
        return self.add_model(name, d * norm, "general")

    def add_specialist_model(self, name: str, direction: np.ndarray,
                             norm: float = 0.85) -> ModelPoint:
        d = np.asarray(direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-10)
        return self.add_model(name, d * norm, "specialist")

    def distance(self, name_a: str, name_b: str) -> float:
        return PoincareBall.distance(self.models[name_a].coords,
                                     self.models[name_b].coords)

    def nearest_model(self, point: np.ndarray, n: int = 1) -> List[Tuple[str, float]]:
        point = PoincareBall.project(np.asarray(point, dtype=np.float64))
        dists = [(name, PoincareBall.distance(point, mp.coords))
                 for name, mp in self.models.items()]
        dists.sort(key=lambda x: x[1])
        return dists[:n]


@dataclass
class RoutingResult:
    """Result of routing a single task."""
    task_id: int
    task_embedding: np.ndarray
    hyperbolic_model: str
    hyperbolic_distance: float
    euclidean_model: str
    euclidean_distance: float
    agree: bool


class TaskRouter:
    """Route tasks to the best model using hyperbolic geometry."""

    def __init__(self, space: CapabilitySpace) -> None:
        self.space = space

    def embed_task(self, constraint_vector: np.ndarray,
                   specialization: float = 0.5) -> np.ndarray:
        cv = np.asarray(constraint_vector, dtype=np.float64)
        cv = cv / (np.linalg.norm(cv) + 1e-10)
        norm = specialization * 0.9
        return PoincareBall.project(cv * norm)

    def route_hyperbolic(self, task_embedding: np.ndarray) -> Tuple[str, float]:
        return self.space.nearest_model(task_embedding, n=1)[0]

    def route_euclidean(self, task_embedding: np.ndarray) -> Tuple[str, float]:
        task = np.asarray(task_embedding, dtype=np.float64)
        dists = [(name, float(np.linalg.norm(task - mp.coords)))
                 for name, mp in self.space.models.items()]
        dists.sort(key=lambda x: x[1])
        return dists[0]

    def route_task(self, task_id: int, constraint_vector: np.ndarray,
                   specialization: float = 0.5) -> RoutingResult:
        emb = self.embed_task(constraint_vector, specialization)
        h_name, h_dist = self.route_hyperbolic(emb)
        e_name, e_dist = self.route_euclidean(emb)
        return RoutingResult(
            task_id=task_id, task_embedding=emb,
            hyperbolic_model=h_name, hyperbolic_distance=h_dist,
            euclidean_model=e_name, euclidean_distance=e_dist,
            agree=(h_name == e_name),
        )

    def route_batch(self, tasks: List[Tuple[int, np.ndarray, float]]
                    ) -> List[RoutingResult]:
        return [self.route_task(tid, cv, spec) for tid, cv, spec in tasks]
