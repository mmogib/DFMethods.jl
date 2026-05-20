```@meta
CurrentModule = DFMethods
```

# DFMethods.jl

**Derivative-free projection methods for constrained nonlinear equations,
plugged into NonlinearSolve.jl.**

Solve

```math
\text{Find } u^* \in X \subset \mathbb{R}^n \text{ such that } F(u^*) = 0,
```

where $X$ is closed convex and $F : \mathbb{R}^n \to \mathbb{R}^n$ is
continuous. The algorithm uses no derivative information of $F$. Precise
convergence assumptions are stated on the [Algorithm](@ref) page.

## What it does

DFMethods.jl provides solvers for nonlinear equations
$F(u) = 0$ with $u$ constrained to a closed convex set $X$. The methods
are *derivative-free* (only function evaluations of $F$ are needed) and
*projection-based* (each iteration produces a feasible iterate by
projecting onto $X$).

The package is organized as a configurable framework: each step of an
outer iteration — inertial extrapolation, search direction, line search,
trial point, iterate update, stopping rule — is an independently
pluggable component. Users can swap in custom rules at each step
without touching the solver core. See [Algorithm](@ref) for the
structure and [Extending](@ref) for the contracts.

## When to use it

DFMethods.jl is appropriate when:

- the problem is a system of equations $F(u) = 0$, not an optimization
  problem (use `Optimization.jl` for the latter);
- a feasibility set $X \subset \mathbb{R}^n$ must be respected at every
  iteration, and $X$ is more general than box bounds;
- evaluating $F$ is feasible but evaluating $\nabla F$ is not (or not
  worthwhile).

For problems outside these conditions, see [Comparisons](@ref) for a
discussion of related packages.

| | Constrained? | Derivative-free? | Approach |
|---|---|---|---|
| NonlinearSolve.jl `DFSane` / `SimpleDFSane` | ✗ | ✓ | spectral residual |
| NonlinearSolve.jl `Broyden` / `Klement` / `LimitedMemoryBroyden` | ✗ | ✓ (secant) | quasi-Newton |
| NonlinearSolve.jl JFNK (Newton + Krylov) | ✗ | ✓ (matrix-free) | Newton–Krylov |
| `NLboxsolve.jl` | box only | ✗ | Newton/QN |
| `ProximalAlgorithms.jl`, `SPGBox.jl` | ✓ | ✗ | minimization |
| **DFMethods.jl** | **any closed convex (incl. ℝⁿ)** | **✓** | **projection-based** |

## Quick navigation

```@contents
Pages = ["quickstart.md", "algorithm.md", "constraints.md",
         "extending.md", "comparisons.md",
         "api.md"]
Depth = 1
```

## Public API in one example

```@setup index
using NonlinearSolve, DFMethods
```

```@example index
F(u, p) = u .- sin.(u)
u0      = ones(100)
prob    = NonlinearProblem(F, u0)            # SciML problem (in-place or out-of-place)
alg     = DFProjection(;                     # configure components
    direction      = SpectralThreeTerm(),    # search direction (or custom)
    linesearch     = ResidualNormBacktrack(),# or ConstantBacktrack / AdaptiveClampedBacktrack
    inertial       = Inertial(0.25),         # or NoInertial()
    iterate_update = SolodovSvaiterProjection(), # or DirectUpdate / HalpernUpdate(β)
    abstol         = 1e-6,
    maxiters       = 2000,
)
sol = solve(prob, alg)                       # NonlinearSolution with retcode, u, resid, stats
sol.retcode
```

Box constraints are passed to the problem (`NonlinearProblem(F, u0; lb,
ub)`); other convex sets use [`ConstrainedNonlinearProblem`](@ref). See
[Constraint Sets](@ref).

## Theoretical lineage

The package's components draw on a body of literature on Solodov–Svaiter-
style projection methods, derivative-free spectral-residual methods, the
Halpern anchoring iteration, and inertial extrapolation for monotone
operators. Each component's docstring cites its originating source.
