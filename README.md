# flux-hyperbolic

Poincaré ball geometry for model capability routing. Pure numpy, zero external dependencies.

## Why Hyperbolic?

In hyperbolic space, distances grow exponentially near the boundary. This naturally models model specialization:

- **General models** live near the center (low norm) — close to everything
- **Specialists** live near the boundary (high norm) — exponentially distant from other specialists
- A math specialist and a code specialist are **much farther apart** in hyperbolic space than Euclidean, reflecting their true capability divergence

This matters for routing: hyperbolic distance captures *qualitative* gaps that Euclidean distance flattens.

## Install

```bash
pip install flux-hyperbolic
```

Or dev mode:

```bash
git clone <repo>
cd flux-hyperbolic-py
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from flux_hyperbolic import CapabilitySpace, TaskRouter, FrechetMean, PoincareBall

# Build a capability space (8D Poincaré ball)
space = CapabilitySpace()

# Add models — generalists near center, specialists near boundary
space.add_general_model("gpt-4o", np.random.randn(8), norm=0.1)
space.add_specialist_model("deepseek-math", np.random.randn(8), norm=0.85)
space.add_specialist_model("claude-code", np.random.randn(8), norm=0.90)

# Route a task to the nearest model
router = TaskRouter(space)
result = router.route_task(
    task_id=1,
    constraint_vector=np.array([2.0, 0.5, 3.0, 1.0, 0.5, 0.2, 1.0, 0.3]),
    specialization=0.8,  # niche task
)

print(f"Hyperbolic: {result.hyperbolic_model} (d={result.hyperbolic_distance:.3f})")
print(f"Euclidean:  {result.euclidean_model} (d={result.euclidean_distance:.3f})")
print(f"Agree: {result.agree}")

# Fleet consensus — hyperbolic centroid
all_coords = [mp.coords for mp in space.models.values()]
centroid = FrechetMean.compute(all_coords)
print(f"Fleet centroid norm: {np.linalg.norm(centroid):.4f}")

# Geometry primitives
d = PoincareBall.distance(np.array([0.1, 0.2]), np.array([0.3, 0.4]))
print(f"Hyperbolic distance: {d:.4f}")
```

## API

### `PoincareBall` — Geometry Operations

| Method | Description |
|--------|-------------|
| `distance(u, v)` | Poincaré distance between two points |
| `project(v, eps=1e-5)` | Clamp point onto ball |
| `mobius_add(u, v)` | Möbius addition |
| `expmap(origin, v)` | Exponential map (tangent → manifold) |
| `logmap(origin, v)` | Logarithmic map (manifold → tangent) |
| `conformal_factor(v)` | λ_v = 2/(1-‖v‖²) |

### `CapabilitySpace` — 8D Model Embedding

8 axes: reasoning_depth, code_generation, math_precision, context_window, speed, creativity, safety, multilingual.

| Method | Description |
|--------|-------------|
| `add_model(name, coords)` | Add model at arbitrary coords |
| `add_general_model(name, direction, norm)` | Add near center |
| `add_specialist_model(name, direction, norm)` | Add near boundary |
| `nearest_model(point, n)` | Find n nearest models |

### `TaskRouter` — Routing

| Method | Description |
|--------|-------------|
| `route_task(id, constraint_vector, specialization)` | Route single task |
| `route_batch(tasks)` | Route batch |
| `embed_task(constraint_vector, specialization)` | Embed task on ball |

### `FrechetMean` — Fleet Consensus

| Method | Description |
|--------|-------------|
| `compute(points, weights)` | Weighted hyperbolic centroid |

## License

MIT
