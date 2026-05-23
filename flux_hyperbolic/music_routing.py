"""
Hyperbolic music routing — embed constraint profiles as points in the
Poincaré ball for infinite genre blending.

Maps AgentPersonality parameters to 8D Poincaré ball coordinates, enabling:
  - Genre distance measurement via hyperbolic geometry
  - Genre blending via Frechet mean
  - Collaborator discovery via nearest-neighbour in hyperbolic space
  - Curvature-controlled creativity boundaries

Works standalone (no flux-tensor-midi dependency) via PersonalityProxy,
and integrates directly when AgentPersonality is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from flux_hyperbolic.geometry import PoincareBall
from flux_hyperbolic.consensus import FrechetMean


# ---------------------------------------------------------------------------
# Personality proxy (works without flux-tensor-midi)
# ---------------------------------------------------------------------------

@dataclass
class PersonalityProxy:
    """Lightweight stand-in for AgentPersonality.

    When flux_tensor_midi is available, ``from_personality()`` extracts
    values directly.  Otherwise, callers can construct these manually.
    """

    name: str
    instrument: str = "unknown"
    midi_channel: int = 0
    midi_program: int = 0
    preferred_intervals: Tuple[int, ...] = (0, 2, 3, 5, 7)
    note_density: float = 2.0
    velocity_range: Tuple[int, int] = (60, 120)
    rest_probability: float = 0.1
    snap_epsilon: float = 0.8
    direction_change_prob: float = 0.4
    sustain_factor: float = 0.5
    octave_range: Tuple[int, int] = (4, 5)
    consensus_weight: float = 0.7

    @classmethod
    def from_personality(cls, p: Any) -> "PersonalityProxy":
        """Construct from an AgentPersonality instance."""
        return cls(
            name=p.name,
            instrument=p.instrument,
            midi_channel=p.midi_channel,
            midi_program=p.midi_program,
            preferred_intervals=tuple(p.preferred_intervals),
            note_density=float(p.note_density),
            velocity_range=tuple(p.velocity_range),
            rest_probability=float(p.rest_probability),
            snap_epsilon=float(p.snap_epsilon),
            direction_change_prob=float(p.direction_change_prob),
            sustain_factor=float(p.sustain_factor),
            octave_range=tuple(p.octave_range),
            consensus_weight=float(p.consensus_weight),
        )


# ---------------------------------------------------------------------------
# Embedding: personality → Poincaré ball point
# ---------------------------------------------------------------------------

# Musical embedding dimensions (8D):
#   0: chromatic_density   — how many distinct intervals (0..1)
#   1: rhythmic_intensity  — note_density normalised
#   2: dynamic_range       — velocity spread (0..1)
#   3: spaciousness        — rest_probability
#   4: timing_tightness    — snap_epsilon
#   5: angularity          — direction_change_prob
#   6: sustain             — sustain_factor
#   7: consensus           — consensus_weight

EMBED_DIM = 8
MAX_NOTE_DENSITY = 8.0  # sensible upper bound for normalisation


def embed_personality(p: PersonalityProxy) -> np.ndarray:
    """Embed a personality proxy as a point in the 8D Poincaré ball.

    Returns
    -------
    np.ndarray
        Shape ``(8,)`` with ``||v|| < 1``.
    """
    chromatic = len(set(p.preferred_intervals)) / 11.0
    rhythmic = min(p.note_density / MAX_NOTE_DENSITY, 1.0)
    dynamic = (p.velocity_range[1] - p.velocity_range[0]) / 127.0
    spacious = p.rest_probability
    timing = p.snap_epsilon
    angular = p.direction_change_prob
    sustain = p.sustain_factor
    consensus = p.consensus_weight

    raw = np.array([
        chromatic, rhythmic, dynamic, spacious,
        timing, angular, sustain, consensus,
    ], dtype=np.float64)

    # Normalise to unit direction, then scale by specialization norm.
    norm = np.linalg.norm(raw)
    if norm < 1e-10:
        return PoincareBall.project(np.zeros(EMBED_DIM))

    # Specialization: more extreme personalities → larger norm → closer to boundary
    # Use a simple heuristic: extremes push toward boundary
    specialization = _specialization_norm(p)

    direction = raw / norm
    point = direction * specialization
    return PoincareBall.project(point)


def _specialization_norm(p: PersonalityProxy) -> float:
    """Estimate specialization level (0..~0.95) from personality parameters.

    High specialization = extreme values in any dimension (very dense,
    very sparse, very angular, very high consensus, etc.)
    """
    chromatic = len(set(p.preferred_intervals)) / 11.0
    rhythmic = min(p.note_density / MAX_NOTE_DENSITY, 1.0)
    dynamic = (p.velocity_range[1] - p.velocity_range[0]) / 127.0

    # Distance from the "average" profile (0.5, 0.25, 0.47, 0.1, 0.8, 0.4, 0.5, 0.7)
    centre = np.array([0.5, 0.25, 0.47, 0.1, 0.8, 0.4, 0.5, 0.7])
    point = np.array([
        chromatic, rhythmic, dynamic, p.rest_probability,
        p.snap_epsilon, p.direction_change_prob, p.sustain_factor,
        p.consensus_weight,
    ])
    eucl_dist = np.linalg.norm(point - centre)
    # Map to [0.1, 0.9]: close to centre → generalist, far → specialist
    spec = 0.1 + 0.8 * min(eucl_dist / 1.0, 1.0)
    return spec


def genre_distance(name_a: str, name_b: str,
                   space: "MusicRoutingSpace") -> float:
    """Hyperbolic distance between two embedded genres in a routing space."""
    return space.distance(name_a, name_b)


def blend_genres(names: List[str], weights: List[float],
                 space: "MusicRoutingSpace") -> np.ndarray:
    """Blend genres via Frechet mean in hyperbolic space.

    Parameters
    ----------
    names : list[str]
        Genre / personality names already embedded in *space*.
    weights : list[float]
        Blend weights (will be normalised to sum to 1).
    space : MusicRoutingSpace
        The routing space containing the embeddings.

    Returns
    -------
    np.ndarray
        The blended point in the Poincaré ball.
    """
    if len(names) != len(weights):
        raise ValueError("names and weights must have the same length")
    points = [space.embedded[name].coords for name in names]
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    return FrechetMean.compute(points, w)


def find_collaborators(
    target: np.ndarray,
    agent_pool: Sequence[PersonalityProxy],
    n: int = 3,
    *,
    space: Optional["MusicRoutingSpace"] = None,
) -> List[Tuple[str, float]]:
    """Find the *n* agents closest in hyperbolic space to a target point.

    If *space* is provided and the agents are already embedded there, those
    embeddings are used.  Otherwise embeddings are computed on the fly.

    Returns
    -------
    list[tuple[str, float]]
        ``(name, distance)`` sorted nearest-first.
    """
    target = PoincareBall.project(np.asarray(target, dtype=np.float64))
    results: List[Tuple[str, float]] = []

    for agent in agent_pool:
        if space is not None and agent.name in space.embedded:
            pt = space.embedded[agent.name].coords
        else:
            pt = embed_personality(agent)
        d = PoincareBall.distance(target, pt)
        results.append((agent.name, d))

    results.sort(key=lambda x: x[1])
    return results[:n]


# ---------------------------------------------------------------------------
# MusicRoutingSpace
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedAgent:
    """An agent embedded in the routing space."""

    name: str
    personality: PersonalityProxy
    coords: np.ndarray

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.coords))

    @property
    def specialization_level(self) -> str:
        n = self.norm
        if n < 0.2:
            return "general"
        elif n < 0.7:
            return "moderate"
        return "specialist"


class MusicRoutingSpace:
    """Routing space for musical agents in the Poincaré ball.

    Embed agents by their constraint profile, measure distances,
    blend genres, and discover collaborators.
    """

    def __init__(self) -> None:
        self.embedded: Dict[str, EmbeddedAgent] = {}

    def embed(self, personality: PersonalityProxy) -> EmbeddedAgent:
        """Embed a personality into the routing space."""
        coords = embed_personality(personality)
        ea = EmbeddedAgent(
            name=personality.name,
            personality=personality,
            coords=coords,
        )
        self.embedded[personality.name] = ea
        return ea

    def embed_personality(self, personality: Any) -> EmbeddedAgent:
        """Embed an AgentPersonality (from flux-tensor-midi) or proxy."""
        if isinstance(personality, PersonalityProxy):
            return self.embed(personality)
        return self.embed(PersonalityProxy.from_personality(personality))

    def distance(self, name_a: str, name_b: str) -> float:
        """Hyperbolic distance between two embedded agents."""
        a = self.embedded[name_a].coords
        b = self.embedded[name_b].coords
        return PoincareBall.distance(a, b)

    def nearest(self, target: np.ndarray, n: int = 3
                ) -> List[Tuple[str, float]]:
        """Find the *n* nearest embedded agents to a target point."""
        target = PoincareBall.project(np.asarray(target, dtype=np.float64))
        results = [
            (name, PoincareBall.distance(target, ea.coords))
            for name, ea in self.embedded.items()
        ]
        results.sort(key=lambda x: x[1])
        return results[:n]

    def blend(self, names: List[str], weights: List[float]
              ) -> np.ndarray:
        """Blend embedded agents via Frechet mean."""
        return blend_genres(names, weights, self)

    def decode_to_personality(
        self,
        point: np.ndarray,
        name: str = "blended",
        *,
        instrument: str = "synth",
        midi_channel: int = 0,
        midi_program: int = 80,
    ) -> PersonalityProxy:
        """Decode a Poincaré ball point back into a PersonalityProxy.

        This is an approximate inverse of ``embed_personality`` — useful
        for turning a blended point into something playable.
        """
        point = np.asarray(point, dtype=np.float64)

        # Rescale from Poincaré direction back to parameter ranges.
        # The raw embedding direction encodes relative weights.
        norm = np.linalg.norm(point)
        if norm < 1e-10:
            # Origin = average profile
            return PersonalityProxy(name=name, instrument=instrument,
                                    midi_channel=midi_channel,
                                    midi_program=midi_program)

        direction = point / norm
        # Scale factor: map direction components to plausible parameter ranges
        chromatic = max(0.0, min(1.0, (direction[0] + 1.0) / 2.0))
        rhythmic = max(0.0, min(1.0, (direction[1] + 1.0) / 2.0))
        dynamic = max(0.0, min(1.0, (direction[2] + 1.0) / 2.0))
        spacious = max(0.0, min(1.0, (direction[3] + 1.0) / 2.0))
        timing = max(0.0, min(1.0, (direction[4] + 1.0) / 2.0))
        angular = max(0.0, min(1.0, (direction[5] + 1.0) / 2.0))
        sustain = max(0.0, min(1.0, (direction[6] + 1.0) / 2.0))
        consensus = max(0.0, min(1.0, (direction[7] + 1.0) / 2.0))

        # Build intervals from chromatic density
        n_intervals = max(1, round(chromatic * 11))
        preferred = tuple(sorted(set(range(n_intervals))))
        if not preferred:
            preferred = (0,)

        return PersonalityProxy(
            name=name,
            instrument=instrument,
            midi_channel=midi_channel,
            midi_program=midi_program,
            preferred_intervals=preferred,
            note_density=max(0.1, rhythmic * MAX_NOTE_DENSITY),
            velocity_range=(
                max(1, int(64 * (1.0 - dynamic * 0.3))),
                min(127, int(64 + 63 * dynamic)),
            ),
            rest_probability=max(0.0, min(1.0, spacious)),
            snap_epsilon=max(0.0, min(1.0, timing)),
            direction_change_prob=max(0.0, min(1.0, angular)),
            sustain_factor=max(0.0, min(1.0, sustain)),
            octave_range=(4, 5),
            consensus_weight=max(0.0, min(1.0, consensus)),
        )

    def all_distances(self) -> Dict[Tuple[str, str], float]:
        """Compute all pairwise hyperbolic distances."""
        names = list(self.embedded.keys())
        result: Dict[Tuple[str, str], float] = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                result[(a, b)] = self.distance(a, b)
        return result
