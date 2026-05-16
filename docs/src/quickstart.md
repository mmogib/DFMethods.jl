```@meta
CurrentModule = DFMethods
```

# Quickstart

DFMethods plugs into NonlinearSolve.jl's standard solve interface. After `using DFMethods`, you call `solve(prob, alg)` exactly as for any SciML algorithm.

## Out-of-place problem

```julia
using NonlinearSolve, DFMethods

# ψ : R^n → R^n  (out-of-place — returns a new vector)
F(u, p) = u .- sin.(u)

n  = 1000
x0 = ones(n)
prob = NonlinearProblem(F, x0)

alg = DFProjection()                # all defaults
sol = solve(prob, alg)

sol.u             # final iterate
sol.retcode       # ReturnCode.Success / MaxIters / Stalled
sol.resid         # final ψ(x*)
sol.stats.nf      # ψ evaluations
sol.stats.nsteps  # outer iterations
```

## In-place problem

The same `NonlinearProblem` constructor accepts in-place mappings; DFMethods handles both automatically:

```julia
function F!(du, u, p)
    @. du = u - sin(u)
    return nothing
end

prob = NonlinearProblem(F!, ones(1000))
sol  = solve(prob, DFProjection())
```

## Adding a constraint

The constraint set is a **field of the algorithm**, not the problem — `NonlinearProblem` is the standard SciML type and stays untouched. See [Constraint Sets](@ref) for the full menu.

```julia
# Box ∩ {Σ x_i ≤ c} — the polyhedral Ω used in Ibrahim 2026
set = CappedBox(-1.0, 1.0, 0.5)   # a, b, c (all scalars)

alg = DFProjection(; set = set)
sol = solve(prob, alg)
```

## Parameters

NonlinearSolve.jl's `p` argument flows through:

```julia
F(u, p) = u .- p
prob = NonlinearProblem(F, [1.0, -1.0], [0.3, -0.2])
sol  = solve(prob, DFProjection())
sol.u ≈ [0.3, -0.2]      # true
```

## Overriding tolerances

`abstol` and `maxiters` accept SciML's standard kwargs and override the algorithm's defaults:

```julia
sol = solve(prob, DFProjection(); abstol = 1e-10, maxiters = 5000)
```

## Choosing a different line search or direction

```julia
alg = DFProjection(;
    direction  = SpectralThreeTerm(; r = 0.05),       # spectral parameter
    linesearch = LSV(; σ = 0.001, ρ = 0.5), # γ_k = min(1, ‖ψ‖)
    inertial   = NoInertial(),              # disable inertia
)
```

See [Algorithm](@ref) for the full list and [Extending](@ref) for how to define your own.

## Inspecting convergence

```julia
using NonlinearSolve.SciMLBase   # for ReturnCode

sol = solve(prob, DFProjection(); maxiters = 2)
sol.retcode == ReturnCode.MaxIters    # true if it ran out of iterations
sol.stats                              # NLStats(nf, njacs, nfactors, nsolve, nsteps)
```

For lower-level access — manual step-by-step iteration through SciML's standard `init` + `step!` + `solve!`:

```julia
using NonlinearSolve, DFMethods
prob  = NonlinearProblem(f, x0)
cache = init(prob, alg)              # CommonSolve.init → DFSciMLCache
while !cache.inner.done
    step!(cache)                     # CommonSolve.step! advances one outer iteration
end
sol = solve!(cache)                  # CommonSolve.solve! finalizes a NonlinearSolution
```

`step!` here is the one reexported through `NonlinearSolve`/`SciMLBase` from `CommonSolve`. DFMethods does **not** export an unqualified `step!` — that would collide with the SciML reexport. If you need the truly internal advance on the inner `DFProjectionCache` (e.g. when building a custom driver around `init_cache`), reach for it qualified: `DFMethods.step!(cache.inner)`.
