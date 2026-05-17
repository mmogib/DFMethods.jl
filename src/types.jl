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
