# stopping_criteria.jl — Pluggable stopping criteria for `DFProjection`.
#
# Three check points per outer iteration of `step!`:
#   - at_w:   after ψ(w_k) is computed
#   - at_z:   after the trial point z_k and ψ(z_k) are computed
#   - at_end: after the projection step → x_{k+1}
#
# Each concrete criterion overrides only the check points it cares about;
# the others fall back to a `(false, :Default)` no-op. Composite via `AnyOf`.

"""
    AbstractStoppingCriterion

Supertype for stopping criteria. Concrete subtypes implement any subset of:

```julia
should_stop_at_w(crit, cache)   -> (Bool, Symbol)   # after ψ(w_k) eval
should_stop_at_z(crit, cache)   -> (Bool, Symbol)   # after ψ(z_k) eval
should_stop_at_end(crit, cache) -> (Bool, Symbol)   # after projection → x_{k+1}
```

The `Symbol` is the `retcode` (`:Success`, `:Stalled`, `:MaxIters`,
`:MaxTime`, `:MaxFEvals`, `:UserStop`, or any user-defined code).
"""
abstract type AbstractStoppingCriterion end

# Default fall-throughs — concrete criteria override the points they care about.
should_stop_at_w(::AbstractStoppingCriterion, cache)   = (false, :Default)
should_stop_at_z(::AbstractStoppingCriterion, cache)   = (false, :Default)
should_stop_at_end(::AbstractStoppingCriterion, cache) = (false, :Default)

# ============================================================================
# Residual-based criteria (fire after ψ(w_k) and ψ(z_k) evals)
# ============================================================================

"""
    AbsResidualTol(abstol)

Stop when ``\\|\\psi(\\cdot)\\| \\le \\text{abstol}`` at either w_k or z_k.
Retcode `:Success`.
"""
struct AbsResidualTol <: AbstractStoppingCriterion
    abstol::Float64
end

function should_stop_at_w(c::AbsResidualTol, cache)
    return norm(cache.ψw) <= c.abstol ? (true, :Success) : (false, :Default)
end

function should_stop_at_z(c::AbsResidualTol, cache)
    return norm(cache.ψz) <= c.abstol ? (true, :Success) : (false, :Default)
end

"""
    RelResidualTol(rtol; abstol=0)

Stop when ``\\|\\psi(\\cdot)\\| \\le \\text{abstol} + \\text{rtol} \\cdot \\|\\psi(x_0)\\|``.
Matches the Dai 2015 / Ibrahim 2026 stopping convention. Retcode `:Success`.
"""
struct RelResidualTol <: AbstractStoppingCriterion
    rtol::Float64
    abstol::Float64
end

RelResidualTol(rtol::Real; abstol::Real = 0.0) =
    RelResidualTol(Float64(rtol), Float64(abstol))

function should_stop_at_w(c::RelResidualTol, cache)
    threshold = c.abstol + c.rtol * cache.ψ0_norm
    return norm(cache.ψw) <= threshold ? (true, :Success) : (false, :Default)
end

function should_stop_at_z(c::RelResidualTol, cache)
    threshold = c.abstol + c.rtol * cache.ψ0_norm
    return norm(cache.ψz) <= threshold ? (true, :Success) : (false, :Default)
end

# ============================================================================
# Stall detectors (fire at end-of-iter)
# ============================================================================

"""
    StepNormTol(xtol)

Stop when ``\\|x_k - x_{k-1}\\| \\le \\text{xtol}`` at the end of an
iteration. Catches stalls without waiting for `MaxIters`. Retcode `:Stalled`.
"""
struct StepNormTol <: AbstractStoppingCriterion
    xtol::Float64
end

function should_stop_at_end(c::StepNormTol, cache)
    cache.k <= 1 && return (false, :Default)   # need both x_k and x_{k-1}
    s = 0.0
    @inbounds for i in eachindex(cache.x)
        s += abs2(cache.x[i] - cache.x_prev[i])
    end
    return sqrt(s) <= c.xtol ? (true, :Stalled) : (false, :Default)
end

"""
    DirectionNormTol(dtol)

Stop when ``\\|d_k\\| \\le \\text{dtol}`` at the end of an iteration.
Retcode `:Stalled`.

A vanishing direction near convergence is not the same as a vanishing
residual — pair this with `AbsResidualTol` (via `AnyOf`) for a complete
picture.
"""
struct DirectionNormTol <: AbstractStoppingCriterion
    dtol::Float64
end

function should_stop_at_end(c::DirectionNormTol, cache)
    cache.k == 0 && return (false, :Default)   # d may be uninitialized
    s = 0.0
    @inbounds for i in eachindex(cache.d)
        s += abs2(cache.d[i])
    end
    return sqrt(s) <= c.dtol ? (true, :Stalled) : (false, :Default)
end

# ============================================================================
# Budget criteria (fire at end-of-iter)
# ============================================================================

"""
    MaxIters(maxiters)

Stop when `cache.k >= maxiters`. Retcode `:MaxIters`.
"""
struct MaxIters <: AbstractStoppingCriterion
    maxiters::Int
end

function should_stop_at_end(c::MaxIters, cache)
    return cache.k >= c.maxiters ? (true, :MaxIters) : (false, :Default)
end

"""
    MaxTime(maxtime)

Stop when elapsed wall-clock time exceeds `maxtime` seconds. Retcode
`:MaxTime`. The reference epoch is `cache.t_start`, set in `init_cache`.
"""
struct MaxTime <: AbstractStoppingCriterion
    maxtime::Float64
end

function should_stop_at_end(c::MaxTime, cache)
    return (time() - cache.t_start) > c.maxtime ? (true, :MaxTime) : (false, :Default)
end

"""
    MaxFEvals(maxevals)

Stop when total function evaluations reach `maxevals`. Retcode `:MaxFEvals`.
"""
struct MaxFEvals <: AbstractStoppingCriterion
    maxevals::Int
end

function should_stop_at_end(c::MaxFEvals, cache)
    return cache.n_evals >= c.maxevals ? (true, :MaxFEvals) : (false, :Default)
end

# ============================================================================
# User-supplied callback
# ============================================================================

"""
    UserStop(f)

User-supplied predicate. `f(cache) -> (stopped::Bool, retcode::Symbol)`.
Called at end-of-iteration. Use for domain-specific criteria — e.g.,
stopping when an application-level metric crosses a threshold, or when
``\\|\\psi(x_k)\\|`` itself drops (which would require evaluating `f` at
`cache.x` inside the callback — an extra `ψ` call per iteration).
"""
struct UserStop{F} <: AbstractStoppingCriterion
    f::F
end

function should_stop_at_end(c::UserStop, cache)
    return c.f(cache)
end

# ============================================================================
# Composite — AnyOf
# ============================================================================

"""
    AnyOf(criteria...)

Composite criterion: stops when any constituent stops. Checked in
declaration order at each check point; the first to fire wins and its
retcode propagates.

```julia
stopping = AnyOf(
    RelResidualTol(1e-8; abstol = 1e-12),
    StepNormTol(1e-10),
    MaxIters(5000),
    MaxTime(60.0),
)
```
"""
struct AnyOf{T<:Tuple} <: AbstractStoppingCriterion
    criteria::T
end

AnyOf(criteria...) = AnyOf(criteria)

function should_stop_at_w(c::AnyOf, cache)
    for crit in c.criteria
        stopped, code = should_stop_at_w(crit, cache)
        if stopped
            return (true, code)
        end
    end
    return (false, :Default)
end

function should_stop_at_z(c::AnyOf, cache)
    for crit in c.criteria
        stopped, code = should_stop_at_z(crit, cache)
        if stopped
            return (true, code)
        end
    end
    return (false, :Default)
end

function should_stop_at_end(c::AnyOf, cache)
    for crit in c.criteria
        stopped, code = should_stop_at_end(crit, cache)
        if stopped
            return (true, code)
        end
    end
    return (false, :Default)
end
