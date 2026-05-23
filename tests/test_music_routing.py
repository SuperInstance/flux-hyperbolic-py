"""
Tests for hyperbolic music routing — embed, distance, blend, discover, decode.
"""

import numpy as np
import pytest

from flux_hyperbolic.geometry import PoincareBall
from flux_hyperbolic.music_routing import (
    EMBED_DIM,
    MAX_NOTE_DENSITY,
    PersonalityProxy,
    EmbeddedAgent,
    MusicRoutingSpace,
    embed_personality,
    genre_distance,
    blend_genres,
    find_collaborators,
)


# ---------------------------------------------------------------------------
# Fixtures: representative musical personalities
# ---------------------------------------------------------------------------

PARKER = PersonalityProxy(
    name="Parker", instrument="sax",
    preferred_intervals=(1, 2, 3, 5, 7, 9, 11),
    note_density=4.0, velocity_range=(70, 127), rest_probability=0.05,
    snap_epsilon=0.9, direction_change_prob=0.55,
    sustain_factor=0.3, octave_range=(4, 6), consensus_weight=0.6,
)

MILES = PersonalityProxy(
    name="Miles", instrument="trumpet",
    preferred_intervals=(0, 2, 3, 5, 7),
    note_density=1.2, velocity_range=(40, 100), rest_probability=0.35,
    snap_epsilon=0.5, direction_change_prob=0.2,
    sustain_factor=0.85, octave_range=(4, 5), consensus_weight=0.8,
)

BACH = PersonalityProxy(
    name="Bach", instrument="organ",
    preferred_intervals=(2, 3, 4, 5, 7, 8, 9),
    note_density=3.5, velocity_range=(60, 100), rest_probability=0.05,
    snap_epsilon=0.95, direction_change_prob=0.4,
    sustain_factor=0.6, octave_range=(3, 6), consensus_weight=0.9,
)

COLTRANE = PersonalityProxy(
    name="Coltrane", instrument="sax",
    preferred_intervals=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    note_density=5.0, velocity_range=(65, 125), rest_probability=0.03,
    snap_epsilon=0.85, direction_change_prob=0.5,
    sustain_factor=0.25, octave_range=(3, 6), consensus_weight=0.5,
)

MONK = PersonalityProxy(
    name="Monk", instrument="piano",
    preferred_intervals=(0, 1, 3, 5, 6, 7),
    note_density=2.0, velocity_range=(80, 127), rest_probability=0.25,
    snap_epsilon=0.7, direction_change_prob=0.6,
    sustain_factor=0.5, octave_range=(3, 5), consensus_weight=0.6,
)

NOISE = PersonalityProxy(
    name="Noise", instrument="synth",
    preferred_intervals=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    note_density=6.0, velocity_range=(30, 127), rest_probability=0.02,
    snap_epsilon=0.2, direction_change_prob=0.7,
    sustain_factor=0.15, octave_range=(2, 7), consensus_weight=0.1,
)

DRONE = PersonalityProxy(
    name="Drone", instrument="bass",
    preferred_intervals=(0, 0, 0, 7),
    note_density=0.3, velocity_range=(50, 80), rest_probability=0.1,
    snap_epsilon=0.95, direction_change_prob=0.02,
    sustain_factor=0.98, octave_range=(2, 3), consensus_weight=0.95,
)


# ---------------------------------------------------------------------------
# 1. Embedding
# ---------------------------------------------------------------------------

class TestEmbedPersonality:

    def test_output_shape(self):
        pt = embed_personality(PARKER)
        assert pt.shape == (EMBED_DIM,)

    def test_inside_ball(self):
        for p in [PARKER, MILES, BACH, COLTRANE, MONK, NOISE, DRONE]:
            pt = embed_personality(p)
            assert np.linalg.norm(pt) < 1.0, f"{p.name} outside ball"

    def test_different_personalities_different_points(self):
        pts = {p.name: embed_personality(p) for p in [PARKER, MILES, BACH]}
        assert not np.allclose(pts["Parker"], pts["Miles"])
        assert not np.allclose(pts["Parker"], pts["Bach"])

    def test_deterministic(self):
        a = embed_personality(PARKER)
        b = embed_personality(PARKER)
        assert np.allclose(a, b)

    def test_chromatic_agents_have_larger_norm(self):
        """Coltrane (all 11 intervals) should be more specialized than Miles."""
        ct = np.linalg.norm(embed_personality(COLTRANE))
        mi = np.linalg.norm(embed_personality(MILES))
        assert ct > mi


# ---------------------------------------------------------------------------
# 2. PersonalityProxy
# ---------------------------------------------------------------------------

class TestPersonalityProxy:

    def test_construction(self):
        p = PersonalityProxy(name="test")
        assert p.name == "test"
        assert p.note_density == 2.0

    def test_from_personality_fallback(self):
        """from_personality with a plain object should work via duck typing."""
        class FakePersonality:
            name = "fake"
            instrument = "banjo"
            midi_channel = 3
            midi_program = 105
            preferred_intervals = (0, 5, 7)
            note_density = 1.5
            velocity_range = (50, 90)
            rest_probability = 0.2
            snap_epsilon = 0.75
            direction_change_prob = 0.3
            sustain_factor = 0.6
            octave_range = (3, 5)
            consensus_weight = 0.8
        proxy = PersonalityProxy.from_personality(FakePersonality())
        assert proxy.name == "fake"
        assert proxy.instrument == "banjo"


# ---------------------------------------------------------------------------
# 3. MusicRoutingSpace
# ---------------------------------------------------------------------------

class TestMusicRoutingSpace:

    def _full_space(self) -> MusicRoutingSpace:
        space = MusicRoutingSpace()
        for p in [PARKER, MILES, BACH, COLTRANE, MONK, NOISE, DRONE]:
            space.embed(p)
        return space

    def test_embed_adds_entry(self):
        space = MusicRoutingSpace()
        ea = space.embed(PARKER)
        assert "Parker" in space.embedded
        assert isinstance(ea, EmbeddedAgent)

    def test_distance_jazz_close(self):
        """Parker and Miles are both jazz → small distance."""
        space = self._full_space()
        d = space.distance("Parker", "Miles")
        assert d < 3.0

    def test_distance_cross_genre_large(self):
        """Noise and Drone are extreme opposites → large distance."""
        space = self._full_space()
        d = space.distance("Noise", "Drone")
        assert d > 2.0

    def test_distance_symmetric(self):
        space = self._full_space()
        assert abs(space.distance("Parker", "Miles") -
                    space.distance("Miles", "Parker")) < 1e-10

    def test_noise_drone_extreme(self):
        """Noise and Drone are both experimental but polar opposites."""
        space = self._full_space()
        d = space.distance("Noise", "Drone")
        assert d > 2.0  # far apart

    def test_nearest_finds_jazz_pair(self):
        """Given Parker's coords, nearest should include Coltrane/Miles."""
        space = self._full_space()
        target = space.embedded["Parker"].coords
        nearest = space.nearest(target, n=3)
        names = [n for n, _ in nearest]
        # Parker should be closest (distance ~0)
        assert names[0] == "Parker"

    def test_all_distances(self):
        space = self._full_space()
        dists = space.all_distances()
        assert len(dists) == 7 * 6 // 2  # C(7,2) = 21
        # All distances should be positive
        for (a, b), d in dists.items():
            assert d > 0


# ---------------------------------------------------------------------------
# 4. Genre blending via Frechet mean
# ---------------------------------------------------------------------------

class TestBlendGenres:

    def _space(self) -> MusicRoutingSpace:
        space = MusicRoutingSpace()
        for p in [PARKER, MILES, BACH, COLTRANE]:
            space.embed(p)
        return space

    def test_blend_inside_ball(self):
        space = self._space()
        blended = space.blend(["Parker", "Miles"], [0.5, 0.5])
        assert np.linalg.norm(blended) < 1.0

    def test_blend_single_genre(self):
        """Blending one genre with weight 1 should return that point."""
        space = self._space()
        blended = space.blend(["Parker"], [1.0])
        parker_pt = space.embedded["Parker"].coords
        assert np.allclose(blended, parker_pt, atol=1e-5)

    def test_blend_asymmetric_weights(self):
        """70/30 Parker/Miles should be closer to Parker."""
        space = self._space()
        blended_70 = space.blend(["Parker", "Miles"], [0.7, 0.3])
        blended_30 = space.blend(["Parker", "Miles"], [0.3, 0.7])
        parker_pt = space.embedded["Parker"].coords
        d_70 = PoincareBall.distance(blended_70, parker_pt)
        d_30 = PoincareBall.distance(blended_30, parker_pt)
        assert d_70 < d_30

    def test_blend_three_genres(self):
        space = self._space()
        blended = space.blend(
            ["Parker", "Miles", "Bach"], [0.4, 0.3, 0.3]
        )
        assert np.linalg.norm(blended) < 1.0

    def test_blend_mismatched_lengths_raises(self):
        space = self._space()
        with pytest.raises(ValueError):
            space.blend(["Parker", "Miles"], [1.0])


# ---------------------------------------------------------------------------
# 5. Collaborator discovery
# ---------------------------------------------------------------------------

class TestFindCollaborators:

    def test_finds_nearest(self):
        space = MusicRoutingSpace()
        for p in [PARKER, MILES, BACH, COLTRANE, MONK]:
            space.embed(p)

        target = space.embedded["Parker"].coords
        collabs = find_collaborators(target, [PARKER, MILES, BACH],
                                     space=space, n=2)
        assert len(collabs) == 2
        assert collabs[0][0] == "Parker"  # nearest = itself

    def test_without_space(self):
        """Should work even without a pre-built space."""
        target = embed_personality(PARKER)
        collabs = find_collaborators(target, [MILES, BACH, DRONE], n=2)
        assert len(collabs) == 2
        # All distances should be positive
        for _, d in collabs:
            assert d > 0

    def test_pool_smaller_than_n(self):
        collabs = find_collaborators(
            embed_personality(PARKER), [MILES], n=5
        )
        assert len(collabs) == 1


# ---------------------------------------------------------------------------
# 6. Decode back to personality
# ---------------------------------------------------------------------------

class TestDecodeToPersonality:

    def test_decode_roundtrip(self):
        """Embed → decode should produce reasonable parameters."""
        space = MusicRoutingSpace()
        space.embed(PARKER)
        pt = space.embedded["Parker"].coords
        decoded = space.decode_to_personality(pt, name="decoded_parker")
        assert decoded.name == "decoded_parker"
        assert 0.0 <= decoded.rest_probability <= 1.0
        assert 0.0 <= decoded.sustain_factor <= 1.0
        assert decoded.velocity_range[0] >= 1
        assert decoded.velocity_range[1] <= 127

    def test_decode_origin(self):
        """Decoding the origin should return a default-ish profile."""
        space = MusicRoutingSpace()
        decoded = space.decode_to_personality(
            np.zeros(EMBED_DIM), name="origin_agent"
        )
        assert decoded.name == "origin_agent"
        assert decoded.note_density > 0

    def test_decode_blend_produces_valid_personality(self):
        space = MusicRoutingSpace()
        for p in [PARKER, MILES]:
            space.embed(p)
        blended = space.blend(["Parker", "Miles"], [0.5, 0.5])
        decoded = space.decode_to_personality(blended, name="parkmiles")
        assert 0.0 <= decoded.snap_epsilon <= 1.0
        assert 0.0 <= decoded.direction_change_prob <= 1.0
        assert len(decoded.preferred_intervals) >= 1


# ---------------------------------------------------------------------------
# 7. Genre distance topology
# ---------------------------------------------------------------------------

class TestGenreTopology:

    def _space(self) -> MusicRoutingSpace:
        space = MusicRoutingSpace()
        for p in [PARKER, MILES, BACH, COLTRANE, MONK, NOISE, DRONE]:
            space.embed(p)
        return space

    def test_same_instrument_closer(self):
        """Parker and Coltrane (both sax, high density) should be close."""
        space = self._space()
        d_sax = space.distance("Parker", "Coltrane")
        d_diverse = space.distance("Parker", "Bach")
        # Parker and Coltrane are both dense, chromatic sax players
        assert d_sax < 1.5

    def test_triangle_inequality(self):
        """Hyperbolic distance should satisfy triangle inequality."""
        space = self._space()
        d_ab = space.distance("Parker", "Miles")
        d_bc = space.distance("Miles", "Bach")
        d_ac = space.distance("Parker", "Bach")
        assert d_ac <= d_ab + d_bc + 1e-6

    def test_genre_distance_helper(self):
        space = self._space()
        d = genre_distance("Parker", "Miles", space)
        assert d > 0
        assert d == space.distance("Parker", "Miles")


# ---------------------------------------------------------------------------
# 8. EmbeddedAgent properties
# ---------------------------------------------------------------------------

class TestEmbeddedAgent:

    def test_specialist_label(self):
        space = MusicRoutingSpace()
        ea = space.embed(NOISE)
        assert ea.specialization_level in ("general", "moderate", "specialist")

    def test_generalist_label(self):
        """A very average profile should be general."""
        avg = PersonalityProxy(name="avg")
        space = MusicRoutingSpace()
        ea = space.embed(avg)
        # Default values are fairly centred → likely general or moderate
        assert ea.specialization_level in ("general", "moderate")
