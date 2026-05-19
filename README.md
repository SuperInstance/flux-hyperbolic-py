# flux-hyperbolic

Poincaré ball geometry for model capability routing. Pure numpy, no external dependencies.

```bash
pip install flux-hyperbolic
```

## How It Works

### Why Hyperbolic Space?

In ordinary (Euclidean) space, the distance between two points is √(Δx² + Δy² + ...). It grows linearly — two points twice as far apart have twice the distance.

In hyperbolic space, distance grows *exponentially* as you approach the boundary of the ball. This has a useful consequence for modeling capabilities:

- **General models** (like GPT-4o) live near the **center** — low norm, short distance to everything
- **Specialists** (like a math-only model) live near the **boundary** — high norm, *exponentially* far from other specialists

A math specialist and a code specialist might look similar in Euclidean space (both "smart models"), but in hyperbolic space their distance explodes — reflecting that they're qualitatively different kinds of intelligence. This makes hyperbolic space a natural fit for routing tasks to the right model.

### The Distance Function

The Poincaré distance between two points u and v on the ball is:

```
d(u, v) = arcosh(1 + 2 · ‖u - v‖² / ((1 - ‖u‖²)(1 - ‖v‖²)))
```

The key terms:
- `‖u‖²` — squared norm of u. As this approaches 1 (the boundary), the denominator `(1 - ‖u‖²)` approaches zero, making the distance blow up.
- A point at the center (norm=0) is equidistant from all boundary points — that's a generalist.
- Two points near the boundary are exponentially far apart — those are specialists.

### Fréchet Mean for Consensus

You can't average points in hyperbolic space the way you do in Euclidean (arithmetic mean doesn't stay on the manifold). Instead, the **Fréchet mean** minimizes the sum of squared hyperbolic distances:

```
argmin_μ  Σ wᵢ · d(μ, xᵢ)²
```

This gives you a hyperbolic centroid — the "average capability profile" of your entire model fleet. It's computed iteratively, projecting back onto the ball at each step.

## What This Module Does

It maps model capabilities onto an 8-dimensional Poincaré ball and uses hyperbolic distance to route tasks to the best model. The 8 axes are: reasoning_depth, code_generation, math_precision, context_window, speed, creativity, safety, multilingual.

## Usage

### Routing Tasks to Models

```python
import numpy as np
from flux_hyperbolic import CapabilitySpace, TaskRouter

# Build a capability space (8D Poincaré ball)
space = CapabilitySpace()

# Generalists near center, specialists near boundary
space.add_general_model("gpt-4o", np.random.randn(8), norm=0.1)
space.add_specialist_model("deepseek-math", np.random.randn(8), norm=0.85)
space.add_specialist_model("claude-code", np.random.randn(8), norm=0.90)

# Route a task
router = TaskRouter(space)
result = router.route_task(
    task_id=1,
    constraint_vector=np.array([2.0, 0.5, 3.0, 1.0, 0.5, 0.2, 1.0, 0.3]),
    specialization=0.8,
)

print(f"Hyperbolic: {result.hyperbolic_model} (d={result.hyperbolic_distance:.3f})")
print(f"Euclidean:  {result.euclidean_model} (d={result.euclidean_distance:.3f})")
print(f"Agree: {result.agree}")
```

When hyperbolic and Euclidean routing disagree, hyperbolic is usually right for specialized tasks — it captures qualitative gaps that Euclidean flattens.

### Fleet Consensus

```python
from flux_hyperbolic import FrechetMean

all_coords = [mp.coords for mp in space.models.values()]
centroid = FrechetMean.compute(all_coords)
print(f"Fleet centroid norm: {np.linalg.norm(centroid):.4f}")
```

### Geometry Primitives

```python
from flux_hyperbolic import PoincareBall

# Distance
d = PoincareBall.distance(np.array([0.1, 0.2]), np.array([0.3, 0.4]))

# Project a point onto the ball (clamp to boundary)
p = PoincareBall.project(np.array([0.99, 0.99]))

# Exponential map: tangent vector → point on manifold
q = PoincareBall.expmap(np.array([0.0, 0.0]), np.array([0.5, 0.3]))

# Logarithmic map: point on manifold → tangent vector
v = PoincareBall.logmap(np.array([0.0, 0.0]), q)

# Conformal factor (distortion at a point)
lam = PoincareBall.conformal_factor(np.array([0.5, 0.3]))
# λ_v = 2 / (1 - ‖v‖²)
```

## API Reference

### `PoincareBall`

| Method | What it does |
|--------|-------------|
| `distance(u, v)` | Poincaré distance |
| `project(v, eps)` | Clamp point onto ball |
| `mobius_add(u, v)` | Möbius addition (hyperbolic translation) |
| `expmap(origin, v)` | Tangent space → manifold |
| `logmap(origin, v)` | Manifold → tangent space |
| `conformal_factor(v)` | Local distortion factor |

### `CapabilitySpace`

| Method | What it does |
|--------|-------------|
| `add_model(name, coords)` | Add at arbitrary coordinates |
| `add_general_model(name, direction, norm)` | Add near center |
| `add_specialist_model(name, direction, norm)` | Add near boundary |
| `nearest_model(point, n)` | Find n nearest models |

### `TaskRouter`

| Method | What it does |
|--------|-------------|
| `route_task(id, constraint_vector, specialization)` | Route single task |
| `route_batch(tasks)` | Route a batch |
| `embed_task(constraint_vector, specialization)` | Embed task on ball |

### `FrechetMean`

| Method | What it does |
|--------|-------------|
| `compute(points, weights)` | Weighted hyperbolic centroid |

## Where to Go Next

| If you want... | Go to |
|----------------|-------|
| The unified library | [flux-lib](../flux-lib-py) |
| CLI constraint checking | [flux-check](../flux-check-py) |
| Genetic expression engine | [flux-genome](../flux-genome-py) |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
