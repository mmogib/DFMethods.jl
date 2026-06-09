```@meta
CurrentModule = DFMethods
```

# Quickstart

DFMethods plugs into NonlinearSolve.jl's standard solve interface. After
`using DFMethods`, `solve(prob, alg)` works exactly as for any SciML
algorithm. The algorithm itself carries no constraint information; the
feasible set is a property of the problem (see [Constraint Sets](@ref)).

```@setup quick
using NonlinearSolve, DFMethods
```

## Out-of-place problem

```@example quick
# F : R^n → R^n  (out-of-place — returns a new vector)
F(u, p) = u .- sin.(u)

n    = 1000
x0   = ones(n)
prob = NonlinearProblem(F, x0)

alg = DFProjection()              # all defaults
sol = solve(prob, alg)
sol.retcode
```

The returned solution exposes the iterate, residual, and statistics:

```@example quick
(retcode = sol.retcode,
 iters   = sol.stats.nsteps,
 nf      = sol.stats.nf,
 resid_norm = sqrt(sum(abs2, sol.resid)))
```

## In-place problem

The same `NonlinearProblem` constructor accepts in-place mappings; DFMethods
handles both forms automatically.

```@example quick
function F!(du, u, p)
    @. du = u - sin(u)
    return nothing
end

sol_ip = solve(NonlinearProblem(F!, ones(1000)), DFProjection())
sol_ip.retcode
```

## Box constraints

Box bounds are passed to the problem via the SciML-standard `lb` / `ub`
keyword arguments. The algorithm requires no change.

```@example quick
prob_box = NonlinearProblem(F, ones(100); lb = fill(-2.0, 100), ub = fill(2.0, 100))
sol_box  = solve(prob_box, DFProjection())
sol_box.retcode
```

## Other convex constraints

For arbitrary closed convex sets, wrap the problem in
[`ConstrainedNonlinearProblem`](@ref):

```@example quick
inner    = NonlinearProblem(F, ones(100))
prob_hs  = ConstrainedNonlinearProblem(inner, HalfSpace(ones(100), 50.0))
sol_hs   = solve(prob_hs, DFProjection())
sol_hs.retcode
```

See [Constraint Sets](@ref) for the full menu of built-in sets.

## Parameters

NonlinearSolve.jl's `p` argument flows through unchanged:

```@example quick
G(u, p) = u .- p
prob_p  = NonlinearProblem(G, [1.0, -1.0], [0.3, -0.2])
sol_p   = solve(prob_p, DFProjection())
sol_p.u ≈ [0.3, -0.2]
```

## Overriding tolerances

`abstol` and `maxiters` are passed through SciML's standard kwargs and
override the algorithm's defaults:

```@example quick
sol_tight = solve(prob, DFProjection(); abstol = 1e-10, maxiters = 5000)
sol_tight.retcode
```

## Choosing a different search direction or line search

```@example quick
alg_alt = DFProjection(;
    direction  = SpectralThreeTerm(; r = 0.05),
    linesearch = ConstantBacktrack(; σ = 1e-4, ρ = 0.5),
    inertial   = NoInertial(),
)
sol_alt = solve(prob, alg_alt)
sol_alt.retcode
```

See [Algorithm](@ref) for the full menu of built-in components and
[Extending](@ref) for the recipes to define your own.

## Observing convergence

The callback API exposes four events during a solve:

```@example quick
hist    = HistoryCallback(; fields = (:k, :F_norm, :α, :n_evals))
alg_cb  = DFProjection(; callbacks = AbstractCallback[hist])
sol_cb  = solve(prob, alg_cb)
length(hist.history), hist.history[end]
```

Custom observers subtype [`AbstractCallback`](@ref) and add a method on
[`on_event!`](@ref); see [Extending](@ref) §7.

## Inspecting termination

```@example quick
using SciMLBase            # for ReturnCode

sol_short = solve(prob, DFProjection(); maxiters = 2, verbose = false)  # we expect (and inspect) a non-Success retcode
sol_short.retcode == ReturnCode.MaxIters
```

The `NLStats` block on `sol.stats` carries iteration and evaluation counts.

## Lower-level access

Manual step-by-step iteration through the SciML `init` / `step!` / `solve!`
contract is available:

```julia
prob  = NonlinearProblem(F, x0)
cache = init(prob, DFProjection())
while !cache.inner.done
    step!(cache)                   # one outer iteration
end
sol = solve!(cache)
```

Here `step!` is `CommonSolve.step!`, reexported through `SciMLBase`.
DFMethods does **not** export an unqualified `step!` — that would collide
with the SciML reexport. The truly internal advance on the inner
[`DFProjectionCache`](@ref) is available qualified:
`DFMethods.step!(cache.inner)`.
