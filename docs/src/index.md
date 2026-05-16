```@meta
CurrentModule = DFMethods
```

# DFMethods.jl

**Derivative-free projection methods for constrained nonlinear equations, plugged into NonlinearSolve.jl.**

Solve

```math
\text{Find } u^* \in X \subset \mathbb{R}^n \text{ such that } \psi(u^*) = 0,
```

where $X$ is closed convex and $\psi : \mathbb{R}^n \to \mathbb{R}^n$ is continuous and satisfies

```math
\psi(x)^\top (x - u^*) \geq 0 \quad \forall x \in \mathbb{R}^n,
```

a property automatically satisfied if $\psi$ is monotone or pseudo-monotone. **No derivatives of $\psi$ are required.**

## Why DFMethods

| | Constrained? | Derivative-free? | CG / projection-based? |
|---|---|---|---|
| `NonlinearSolve.SimpleDFSane` | ✗ | ✓ | ✗ |
| `NLboxsolve.jl` | box only | ✗ | ✗ |
| `ProximalAlgorithms.jl`, `SPGBox.jl` | ✓ | ✗ | ✗ (minimization) |
| **DFMethods.jl** | **general convex** | **✓** | **✓** |

DFMethods.jl is the first Julia implementation of the Solodov–Svaiter hyperplane-projection family with derivative-free CG-style search directions.

## Quick navigation

```@contents
Pages = ["quickstart.md", "algorithm.md", "constraints.md", "extending.md", "api.md"]
Depth = 1
```

## Public API in one paragraph

```julia
using NonlinearSolve, DFMethods

prob = NonlinearProblem(F, u0, p)                  # SciML problem (in-place or out-of-place)
alg  = DFProjection(;                              # configure components
    direction  = SpectralThreeTerm(),                        # search direction (or custom)
    linesearch = LSII(),                           # one of LSI..LSVII (or custom)
    inertial   = Inertial(0.25),                   # or NoInertial()
    set        = BoxSet(lower, upper),             # or RealSpace / HalfSpace / CappedBox / UserSet
    abstol     = 1e-6,
    maxiters   = 2000,
)
sol = solve(prob, alg)                             # NonlinearSolution with retcode, u, resid, stats
```

## Reference

Ibrahim, A. H., Alshahrani, M., & Al-Homidan, S. (2026). *A Unified Derivative-Free Projection Framework for Convex-Constrained Nonlinear Equations.* Journal of Optimization Theory and Applications, **208**:11. [DOI:10.1007/s10957-025-02826-x](https://doi.org/10.1007/s10957-025-02826-x).
