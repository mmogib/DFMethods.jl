# nonlinearsolve_integration.jl — Plug DFProjection into the SciML
# solve(prob, alg) workflow via CommonSolve's init / solve! contract.
#
# CommonSolve defines:
#
#     solve(prob, args...; kwargs...) = solve!(init(prob, args...; kwargs...))
#
# So `solve(prob, alg; kw...)` calls `init(prob, alg; kw...)` (which must
# return a cache) and then `solve!(cache)` (which must return the solution).
#
# Convention: `AbstractDFProjectionAlgorithm` already subtypes
# `SciMLBase.AbstractNonlinearAlgorithm` (declared in types.jl), so the
# only hooks we add here are the two CommonSolve methods plus a thin
# `DFSciMLCache` wrapper that carries the problem (needed for
# `build_solution`) alongside the inner Phase 2 cache.

using CommonSolve

# ============================================================================
# Wrap NonlinearProblem.f into the out-of-place F(x) form Phase 2 expects
# ============================================================================
#
# Returns a closure `F(x) -> Vector` regardless of whether `prob.f` is
# the in-place form `f!(du, u, p)` or the out-of-place form `f(u, p)`.
# Captures `prob.p` and (in the in-place case) a reusable internal buffer.
#
# The in-place form returns a *copy* of the internal buffer each call so
# downstream callers can hold the result across non-consecutive ψ-calls.
# A polish item is threading an in-place `F!(out, x)` all the way through
# the inner cache to eliminate the copy.
function _wrap_problem_F(prob::SciMLBase.NonlinearProblem)
    f = prob.f
    p = prob.p
    u0 = prob.u0
    if SciMLBase.isinplace(f)
        out_buf = Vector{eltype(u0)}(undef, length(u0))
        return function (x)
            f(out_buf, x, p)
            return copy(out_buf)
        end
    else
        return function (x)
            return f(x, p)
        end
    end
end

# ============================================================================
# DFMethods retcode → SciMLBase.ReturnCode
# ============================================================================

@inline function _to_sciml_retcode(rc::Symbol)
    rc === :Success            ? SciMLBase.ReturnCode.Success    :
    rc === :MaxIters           ? SciMLBase.ReturnCode.MaxIters   :
    rc === :MaxTime            ? SciMLBase.ReturnCode.Terminated :
    rc === :MaxFEvals          ? SciMLBase.ReturnCode.Terminated :
    rc === :Stalled            ? SciMLBase.ReturnCode.Stalled    :
    rc === :LineSearchFailed   ? SciMLBase.ReturnCode.Failure    :
    rc === :DegenerateResidual ? SciMLBase.ReturnCode.Stalled    :
                                 SciMLBase.ReturnCode.Default
end

# ============================================================================
# Effective algorithm with SciML kwarg overrides
# ============================================================================

function _alg_with_overrides(alg::DFProjection, abstol::Real, maxiters::Int)
    if abstol == alg.abstol && maxiters == alg.maxiters
        return alg
    end
    return DFProjection(;
        direction     = alg.direction,
        linesearch    = alg.linesearch,
        inertial      = alg.inertial,
        set           = alg.set,
        abstol        = Float64(abstol),
        maxiters      = maxiters,
        ζ             = alg.ζ,
        inner_maxiter = alg.inner_maxiter,
        maxbt         = alg.maxbt,
    )
end

# ============================================================================
# SciML-style cache wrapper
# ============================================================================

"""
    DFSciMLCache

Wrapper cache returned by `CommonSolve.init(prob, alg; …)`. Carries
the original `NonlinearProblem`, the user-facing algorithm, and the
inner Phase 2 `DFProjectionCache`. `CommonSolve.solve!` drives the inner
cache to termination and packages the result as a `NonlinearSolution`.
"""
mutable struct DFSciMLCache{Prob<:SciMLBase.NonlinearProblem,
                            Alg<:DFProjection,
                            Inner<:DFProjectionCache}
    prob::Prob
    alg::Alg
    inner::Inner
end

# ============================================================================
# CommonSolve.init dispatch
# ============================================================================

"""
    CommonSolve.init(prob::NonlinearProblem, alg::DFProjection; abstol, maxiters, kwargs...)
        -> DFSciMLCache

Build a cache for `solve(prob, alg; …)`. Accepts SciML's standard
`abstol` and `maxiters` kwargs (overriding `alg.abstol` / `alg.maxiters`);
other kwargs are absorbed without effect (Phase 3 polish: route
`verbose`, `callback`, etc.).
"""
function CommonSolve.init(prob::SciMLBase.NonlinearProblem, alg::DFProjection;
                          abstol::Real  = alg.abstol,
                          maxiters::Int = alg.maxiters,
                          kwargs...)
    F       = _wrap_problem_F(prob)
    x0      = collect(Float64, prob.u0)
    alg_eff = _alg_with_overrides(alg, abstol, maxiters)
    inner   = init_cache(F, x0, alg_eff)
    return DFSciMLCache(prob, alg, inner)
end

# ============================================================================
# CommonSolve.solve! dispatch
# ============================================================================

"""
    CommonSolve.solve!(cache::DFSciMLCache) -> NonlinearSolution

Drive the inner cache to termination, then build and return a
`SciMLBase.NonlinearSolution`.
"""
function CommonSolve.solve!(cache::DFSciMLCache)
    inner = cache.inner
    while !inner.done
        step!(inner)
    end
    # Finalize residual if the loop hit max-iters or line-search failure
    # without setting it via an early-return branch.
    if isnan(inner.resid)
        Fx_final = inner.F(inner.x)
        s = 0.0
        @inbounds for i in eachindex(Fx_final)
            s += Fx_final[i] * Fx_final[i]
        end
        inner.resid    = sqrt(s)
        inner.n_evals += 1
    end
    resid_vec = inner.F(inner.x)
    stats = SciMLBase.NLStats(inner.n_evals + 1, 0, 0, 0, inner.k)
    return SciMLBase.build_solution(cache.prob, cache.alg,
                                    inner.x, resid_vec;
                                    retcode = _to_sciml_retcode(inner.retcode),
                                    stats   = stats)
end

# ============================================================================
# CommonSolve.step! dispatch — manual single-iteration advance
# ============================================================================

"""
    CommonSolve.step!(cache::DFSciMLCache) -> cache

Advance the inner cache by one outer iteration of the algorithm. No-op
if the inner cache has already terminated. Returns the wrapper cache so
calls chain.

Usage:

```julia
using NonlinearSolve, DFMethods
cache = init(prob, alg)
while !cache.inner.done
    step!(cache)
end
sol = solve!(cache)
```

For batched runs, `solve(prob, alg)` (or equivalently `solve!(init(prob, alg))`)
is the higher-level entry point that handles the loop internally.
"""
function CommonSolve.step!(cache::DFSciMLCache)
    cache.inner.done && return cache
    DFMethods.step!(cache.inner)
    return cache
end
