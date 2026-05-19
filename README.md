# DFMethods

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mmogib.github.io/DFMethods.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mmogib.github.io/DFMethods.jl/dev/)
[![Build Status](https://github.com/mmogib/DFMethods.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mmogib/DFMethods.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/mmogib/DFMethods.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mmogib/DFMethods.jl)

**Derivative-free projection methods for constrained nonlinear equations, integrated with NonlinearSolve.jl.**

Solve

$$\text{Find } u^* \in X \subset \mathbb{R}^n \text{ such that } F(u^*) = 0,$$

where $X$ is closed convex and $F$ is continuous and (pseudo-)monotone. **No derivatives required.**

## Status

A generic derivative-free projection framework with pluggable search direction, line search, iterate-update strategy, inertial rule, constraint set, and callbacks. The algorithm itself is problem-agnostic; the feasible set is a property of the problem, not the algorithm. See [`CHANGELOG.md`](CHANGELOG.md) for the per-version capability surface.

## Where it sits in the Julia ecosystem

| | Constrained? | Derivative-free? | CG / projection-based? |
|---|---|---|---|
| NonlinearSolve.jl (`SimpleDFSane`) | ✗ | ✓ | ✗ (spectral residual) |
| NLboxsolve.jl | box only | ✗ | ✗ |
| ProximalAlgorithms.jl / SPGBox.jl | ✓ | ✗ | ✗ (minimization, not $F(x)=0$) |
| **DFMethods.jl** | **general convex** | **✓** | **✓ (projection-family)** |

First Julia implementation of the Solodov–Svaiter hyperplane-projection family with derivative-free CG-style search directions, plus alternative iterate-update strategies (direct projection, Halpern anchoring), and fully pluggable line search / inertia / constraint set / callbacks.

## Install

From the Julia General registry:

```julia
using Pkg
Pkg.add("DFMethods")
```

Or, equivalently, from the REPL's Pkg mode:

```
] add DFMethods
```

To track the latest unreleased changes directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/mmogib/DFMethods.jl")
```

## Quickstart

```julia
using NonlinearSolve, DFMethods

# F : R^n → R^n  (out-of-place; in-place f!(du, u, p) also supported)
F(u, p) = u .- p

# Unconstrained
prob = NonlinearProblem(F, [1.0, -1.0], [0.3, -0.2])
sol  = solve(prob, DFProjection())
sol.u             # ≈ [0.3, -0.2]
sol.retcode       # ReturnCode.Success
sol.stats.nf      # number of F evaluations
sol.stats.nsteps  # outer iterations
```

Box constraints flow through SciML's standard `lb` / `ub` kwargs:

```julia
prob_box = NonlinearProblem(F, [1.0, -1.0], [0.3, -0.2];
                            lb = [-1.0, -1.0], ub = [1.0, 1.0])
sol_box  = solve(prob_box, DFProjection())
```

For arbitrary closed convex sets, wrap with `ConstrainedNonlinearProblem`:

```julia
inner = NonlinearProblem(F, [1.0, -1.0], [0.3, -0.2])
prob_hs = ConstrainedNonlinearProblem(inner, HalfSpace([1.0, 1.0], 0.5))
sol_hs  = solve(prob_hs, DFProjection())
```

The same `DFProjection()` instance solves all three problems — the constraint set lives on the problem.

See the [Quickstart](https://mmogib.github.io/DFMethods.jl/stable/quickstart/) and [Extending](https://mmogib.github.io/DFMethods.jl/stable/extending/) pages of the docs for callbacks, custom directions / line searches, and lower-level access.

## Algorithm

The package implements the unified derivative-free projection framework analysed in

> Ibrahim, A. H., Alshahrani, M., & Al-Homidan, S. (2026). *A Unified Derivative-Free Projection Framework for Convex-Constrained Nonlinear Equations.* Journal of Optimization Theory and Applications, **208**:11. <https://doi.org/10.1007/s10957-025-02826-x>

One outer iteration: inertial extrapolation → derivative-free search direction → backtracking line search → trial point → iterate update (Solodov–Svaiter hyperplane projection, direct projection, or Halpern anchoring).

## Pluggable components

| Component | Abstract type | Built-in instances |
|---|---|---|
| Search direction | `AbstractSearchDirection` | `SpectralThreeTerm` |
| Line search | `LineSearch.AbstractLineSearchAlgorithm` | `ConstantBacktrack`, `ResidualNormBacktrack`, `AdaptiveClampedBacktrack` |
| Iterate update | `AbstractIterateUpdate` | `SolodovSvaiterProjection`, `DirectUpdate`, `HalpernUpdate` |
| Inertial rule | `AbstractInertialRule` | `Inertial(θ)`, `NoInertial` |
| Constraint set (on the problem) | `AbstractConstraintSet` | `RealSpace`, `BoxSet`, `HalfSpace`, `Intersection`, `CappedBox`, `UserSet` |
| Stopping / observers | `AbstractCallback` | `AbsResidualTol`, `RelResidualTol`, `MaxIters`, `MaxTime`, `MaxFEvals`, `HistoryCallback`, `LoggingCallback`, … |

Each is a small struct with one required method; see the [Extending](https://mmogib.github.io/DFMethods.jl/stable/extending/) page of the docs.

## Citation

If you use DFMethods.jl in research, please cite:

```bibtex
@article{ibrahim_unified_2026,
  title   = {A Unified Derivative-Free Projection Framework for Convex-Constrained Nonlinear Equations},
  author  = {Ibrahim, Abdulkarim Hassan and Alshahrani, Mohammed and Al-Homidan, Suliman},
  journal = {Journal of Optimization Theory and Applications},
  volume  = {208},
  number  = {11},
  year    = {2026},
  doi     = {10.1007/s10957-025-02826-x},
}
```

## License

MIT
