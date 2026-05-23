"""
FLUX Hyperbolic — Poincaré Ball Geometry for Model Capability Routing.

Hyperbolic geometry router that maps model capabilities onto a Poincaré ball,
where distance naturally captures specialization hierarchies.
"""

from flux_hyperbolic.geometry import PoincareBall
from flux_hyperbolic.capability import CapabilitySpace, TaskRouter, ModelPoint, RoutingResult
from flux_hyperbolic.consensus import FrechetMean
from flux_hyperbolic.music_routing import (
    MusicRoutingSpace,
    PersonalityProxy,
    EmbeddedAgent,
    embed_personality,
    genre_distance,
    blend_genres,
    find_collaborators,
    EMBED_DIM,
    MAX_NOTE_DENSITY,
)

__all__ = [
    "PoincareBall",
    "CapabilitySpace",
    "TaskRouter",
    "ModelPoint",
    "RoutingResult",
    "FrechetMean",
    "MusicRoutingSpace",
    "PersonalityProxy",
    "EmbeddedAgent",
    "embed_personality",
    "genre_distance",
    "blend_genres",
    "find_collaborators",
    "EMBED_DIM",
    "MAX_NOTE_DENSITY",
]
