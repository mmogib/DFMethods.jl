# types.jl — Abstract algorithm supertype.

"""
    AbstractDFProjection

Supertype for all derivative-free projection algorithms in DFMethods.
Concrete subtypes (e.g., `DFProjection` in `algorithm.jl`) wire together
a search direction, a line search, an inertial rule, and a constraint set.

Subtypes `SciMLBase.AbstractNonlinearAlgorithm`, so every concrete
algorithm participates in `solve(prob::NonlinearProblem, alg; kwargs...)`
out of the box.

Convergence assumptions (e.g., monotonicity of ``\\psi``, closed convex
``X``) are documented in [`docs/src/algorithm.md`](@ref). The library
does not enforce them at runtime — algorithms iterate regardless and
either converge or terminate via a stopping criterion.
"""
abstract type AbstractDFProjection <: SciMLBase.AbstractNonlinearAlgorithm end

"""
    init_state(component, prob, x0, alg) -> state

Allocate per-solve state for a pluggable algorithm `component` (search
direction, line search, iterate-update strategy, stopping criterion).
The returned `state` is held on `DFProjectionCache` in the slot matching
the component's role. Components that are stateless (e.g. `SpectralThreeTerm`)
return `nothing`; components that need per-iteration scratch (e.g. the
default Solodov–Svaiter iterate-update strategy with its Dykstra buffers)
return a small struct holding their allocations.

Default: `nothing`. Concrete components override.

This is an internal helper (not exported). The `DFProjectionCache`
constructor calls it once per pluggable component during `init_cache`.
"""
function init_state end

"""
    AbstractIterateUpdate

Supertype for the iterate-update strategy: given the inputs
`(w, d, α, z, Fw, Fz, set, k, ζ, inner_maxiter, state)`, produce
`x_{k+1}`. Concrete subtypes implement

```julia
update_iterate!(x_new, rule::AbstractIterateUpdate, ctx)
```

writing into `x_new` and returning it. `ctx` is a NamedTuple; see
[`SolodovSvaiterProjection`](@ref), [`DirectUpdate`](@ref), and
[`HalpernUpdate`](@ref) for built-in strategies.

The strategy may declare its per-solve state via `init_state(rule, prob, x0, alg)`.
The state is held on `DFProjectionCache.iterate_update_state` and surfaced
to `update_iterate!` via `ctx.state`.
"""
abstract type AbstractIterateUpdate end

"""
    update_iterate!(x_new, rule::AbstractIterateUpdate, ctx) -> x_new

In-place: write the next iterate into `x_new` per `rule`'s logic.
`ctx` is a NamedTuple with fields `w`, `d`, `α`, `z`, `Fw`, `Fz`, `set`,
`k`, `ζ`, `inner_maxiter`, `state`. See [`AbstractIterateUpdate`](@ref).
"""
function update_iterate! end

"""
    AbstractCallback

Supertype for solve-time callbacks. Implement
`on_event!(cb, cache, event::Symbol)` to observe state at specific
lifecycle events. Default behavior: observe-only (return `nothing`).

Stopping criteria are a special case — see [`AbstractStoppingCriterion`](@ref),
which subtypes `AbstractCallback` and returns `(stopped::Bool, retcode::Symbol)`
from `on_event!` to signal termination.

Events fired in v0.2:

| Event | Where it fires |
|---|---|
| `:initialize` | Once, at end of `init_cache` |
| `:post_linesearch` | After the line search; `cache.Fz` is fresh |
| `:post_iter` | After the state shift; `cache.k` incremented |
| `:terminate` | Once, on any termination |

The minimal 4-event set is intentional. More events (`:pre_iter`,
`:post_direction`, `:post_iterate_update`) may be added in a future
release if real need emerges.
"""
abstract type AbstractCallback end

"""
    on_event!(cb, cache, event::Symbol)

Called by the algorithm at each instrumented lifecycle event. For
`AbstractCallback` subtypes that observe only (e.g. `HistoryCallback`,
`LoggingCallback`), return value is ignored. For
`AbstractStoppingCriterion` subtypes, return `(stopped::Bool, retcode::Symbol)`.

Default (observer): `nothing`.
"""
function on_event! end

on_event!(::AbstractCallback, cache, event::Symbol) = nothing

"""
    ConstrainedNonlinearProblem(inner::NonlinearProblem, set::AbstractConstraintSet)
    ConstrainedNonlinearProblem(f, u0, p = NullParameters();
                                set::AbstractConstraintSet,
                                lb = nothing, ub = nothing, kwargs...)

Wraps a `SciMLBase.NonlinearProblem` with a feasibility set the algorithm
must respect. The constraint becomes part of the *problem* (where it
belongs), not the *algorithm* — so one `DFProjection` instance solves
many problems with different feasibility sets.

Use this for non-box constraints (`HalfSpace`, `CappedBox`, `Intersection`,
`UserSet`). For pure box constraints, prefer `NonlinearProblem(f, u0;
lb, ub)` directly — SciMLBase's native support.
"""
struct ConstrainedNonlinearProblem{P, S}
    inner::P
    set::S
end

function ConstrainedNonlinearProblem(f, u0::AbstractVector;
                                       set,
                                       p = SciMLBase.NullParameters(),
                                       lb = nothing, ub = nothing,
                                       kwargs...)
    inner = SciMLBase.NonlinearProblem(f, u0, p; lb = lb, ub = ub, kwargs...)
    return ConstrainedNonlinearProblem(inner, set)
end
