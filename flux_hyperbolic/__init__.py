"""
FLUX Hyperbolic — Poincaré Ball Geometry for Model Capability Routing.

Hyperbolic geometry router that maps model capabilities onto a Poincaré ball,
where distance naturally captures specialization hierarchies.
"""

from flux_hyperbolic.geometry import PoincareBall
from flux_hyperbolic.capability import CapabilitySpace, TaskRouter, ModelPoint, RoutingResult
from flux_hyperbolic.consensus import FrechetMean

__all__ = [
    "PoincareBall",
    "CapabilitySpace",
    "TaskRouter",
    "ModelPoint",
    "RoutingResult",
    "FrechetMean",
]
