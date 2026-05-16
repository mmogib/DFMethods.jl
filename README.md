# DFMethods

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mmogib.github.io/DFMethods.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mmogib.github.io/DFMethods.jl/dev/)
[![Build Status](https://github.com/mmogib/DFMethods.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mmogib/DFMethods.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/mmogib/DFMethods.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mmogib/DFMethods.jl)

**Derivative-free projection methods for constrained nonlinear equations, integrated with NonlinearSolve.jl.**

Solve

$$\text{Find } u^* \in X \subset \mathbb{R}^n \text{ such that } \psi(u^*) = 0,$$

where $X$ is closed convex and $\psi$ is continuous and (pseudo-)monotone. **No derivatives required.**

## Status

v0.1.0 — first registered release. Public API stable. Documentation and full benchmark suite in progress.

## Where it sits in the Julia ecosystem

| | Constrained? | Derivative-free? | CG / projection-based? |
|---|---|---|---|
| NonlinearSolve.jl (`SimpleDFSane`) | ✗ | ✓ | ✗ (spectral residual) |
| NLboxsolve.jl | box only | ✗ | ✗ |
| ProximalAlgorithms.jl / SPGBox.jl | ✓ | ✗ | ✗ (minimization, not $\psi(x)=0$) |
| **DFMethods.jl** | **general convex** | **✓** | **✓ (Solodov–Svaiter)** |

First Julia implementation of the Solodov–Svaiter hyperplane-projection family with derivative-free CG-style search directions and pluggable line search / inertia / constraint set.

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

# ψ : R^n → R^n  (out-of-place; in-place f!(du, u, p) also supported)
F(u, p) = u .- p

# NonlinearProblem with target p = [0.3, -0.2]
prob = NonlinearProblem(F, [1.0, -1.0], [0.3, -0.2])

# Constrained to the box [-1, 1]²
alg = DFProjection(; set = BoxSet([-1.0, -1.0], [1.0, 1.0]))

sol = solve(prob, alg)
sol.u             # ≈ [0.3, -0.2]
sol.retcode       # ReturnCode.Success
sol.stats.nf      # number of ψ evaluations
sol.stats.nsteps  # outer iterations
```

See `examples/extending.jl` for how to define your own search direction and line search.

## Algorithm

Implements **UIDFPAF** — the *Unified Inertial Derivative-Free Projection Algorithmic Framework* from

> Ibrahim, A. H., Alshahrani, M., & Al-Homidan, S. (2026). *A Unified Derivative-Free Projection Framework for Convex-Constrained Nonlinear Equations.* Journal of Optimization Theory and Applications, **208**:11. <https://doi.org/10.1007/s10957-025-02826-x>

One outer iteration: inertial extrapolation → derivative-free direction → backtracking line search → trial-point check → hyperplane projection → approximate projection onto $X \cap H_k$.

## Pluggable components

| Component | Abstract type | Built-in instances |
|---|---|---|
| Search direction | `AbstractSearchDirection` | `SpectralThreeTerm` |
| Line search | `AbstractDFLineSearch` | `LSI`, `LSII`, `LSIII`, `LSIV`, `LSV`, `LSVI`, `LSVII` |
| Inertial rule | `AbstractInertialRule` | `Inertial(θ)`, `NoInertial` |
| Constraint set | `AbstractConstraintSet` | `RealSpace`, `BoxSet`, `HalfSpace`, `Intersection`, `CappedBox`, `UserSet` |

Each is a small struct with one required method; see the *Extending* page of the docs.

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
