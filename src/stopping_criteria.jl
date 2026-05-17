# stopping_criteria.jl — Stopping criteria as callback subtypes.
#
# Each concrete criterion subtypes `AbstractStoppingCriterion <: AbstractCallback`
# and implements `on_event!(crit, cache, event::Symbol) -> (Bool, Symbol)`.
# At the events the criterion cares about, it returns `(true, :Retcode)` to
# signal termination; otherwise `(false, :Default)`.
#
# v0.2 event set: `:initialize`, `:post_linesearch`, `:post_iter`, `:terminate`.
# Residual-based criteria fire at :post_linesearch (cache.Fz is fresh, before
# the projection step); iteration-budget criteria fire at :post_iter (cache.k
# is incremented).

"""
    AbstractStoppingCriterion <: AbstractCallback

Supertype for stopping criteria. Concrete subtypes implement
`on_event!(crit, cache, event::Symbol) -> (Bool, Symbol)` for the events
they care about.

The default fallback returns `(false, :Default)` — i.e., never stops.
Concrete criteria override `on_event!` for their relevant event(s).
"""
abstract type AbstractStoppingCriterion <: AbstractCallback end

# Stateless by default.
init_state(::AbstractStoppingCriterion, prob, x0, alg) = nothing

# Default: never stops. Concrete criteria override.
on_event!(::AbstractStoppingCriterion, cache, event::Symbol) = (false, :Default)

# ============================================================================
# Residual-based criteria (fire at :post_linesearch on cache.Fz)
# ============================================================================

"""
    AbsResidualTol(abstol)

Stop when ``\\|F(z_k)\\| \\le \\text{abstol}`` (post line search).
Retcode `:Success`.
"""
struct AbsResidualTol <: AbstractStoppingCriterion
    abstol::Float64
end

function on_event!(c::AbsResidualTol, cache, event::Symbol)
    event === :post_linesearch || return (false, :Default)
    return norm(cache.Fz) <= c.abstol ? (true, :Success) : (false, :Default)
end

"""
    RelResidualTol(rtol; abstol=0)

Stop when ``\\|F(z_k)\\| \\le \\text{abstol} + \\text{rtol} \\cdot \\|F(x_0)\\|``
(post line search). Retcode `:Success`.
"""
struct RelResidualTol <: AbstractStoppingCriterion
    rtol::Float64
    abstol::Float64
end

RelResidualTol(rtol::Real; abstol::Real = 0.0) =
    RelResidualTol(Float64(rtol), Float64(abstol))

function on_event!(c::RelResidualTol, cache, event::Symbol)
    event === :post_linesearch || return (false, :Default)
    threshold = c.abstol + c.rtol * cache.F0_norm
    return norm(cache.Fz) <= threshold ? (true, :Success) : (false, :Default)
end

# ============================================================================
# Stall detectors (fire at :post_iter)
# ============================================================================

"""
    StepNormTol(xtol)

Stop when ``\\|x_k - x_{k-1}\\| \\le \\text{xtol}`` at end of iteration.
Catches stalls. Retcode `:Stalled`.
"""
struct StepNormTol <: AbstractStoppingCriterion
    xtol::Float64
end

function on_event!(c::StepNormTol, cache, event::Symbol)
    event === :post_iter || return (false, :Default)
    cache.k <= 1 && return (false, :Default)   # need both x_k and x_{k-1}
    s = 0.0
    @inbounds for i in eachindex(cache.x)
        s += abs2(cache.x[i] - cache.x_prev[i])
    end
    return sqrt(s) <= c.xtol ? (true, :Stalled) : (false, :Default)
end

"""
    DirectionNormTol(dtol)

Stop when ``\\|d_k\\| \\le \\text{dtol}``. Retcode `:Stalled`.
Pair with `AbsResidualTol` (via `AnyOf`) for a complete picture —
a vanishing direction is not the same as a vanishing residual.
"""
struct DirectionNormTol <: AbstractStoppingCriterion
    dtol::Float64
end

function on_event!(c::DirectionNormTol, cache, event::Symbol)
    event === :post_iter || return (false, :Default)
    cache.k == 0 && return (false, :Default)
    s = 0.0
    @inbounds for i in eachindex(cache.d)
        s += abs2(cache.d[i])
    end
    return sqrt(s) <= c.dtol ? (true, :Stalled) : (false, :Default)
end

# ============================================================================
# Budget criteria (fire at :post_iter)
# ============================================================================

"""
    MaxIters(maxiters)

Stop when `cache.k >= maxiters`. Retcode `:MaxIters`.
"""
struct MaxIters <: AbstractStoppingCriterion
    maxiters::Int
end

function on_event!(c::MaxIters, cache, event::Symbol)
    event === :post_iter || return (false, :Default)
    return cache.k >= c.maxiters ? (true, :MaxIters) : (false, :Default)
end

"""
    MaxTime(maxtime)

Stop when elapsed wall-clock time exceeds `maxtime` seconds. Retcode `:MaxTime`.
"""
struct MaxTime <: AbstractStoppingCriterion
    maxtime::Float64
end

function on_event!(c::MaxTime, cache, event::Symbol)
    event === :post_iter || return (false, :Default)
    return (time() - cache.t_start) > c.maxtime ? (true, :MaxTime) : (false, :Default)
end

"""
    MaxFEvals(maxevals)

Stop when total function evaluations reach `maxevals`. Retcode `:MaxFEvals`.
"""
struct MaxFEvals <: AbstractStoppingCriterion
    maxevals::Int
end

function on_event!(c::MaxFEvals, cache, event::Symbol)
    event === :post_iter || return (false, :Default)
    return cache.n_evals >= c.maxevals ? (true, :MaxFEvals) : (false, :Default)
end

# ============================================================================
# User-supplied callback
# ============================================================================

"""
    UserStop(f)

User-supplied predicate. `f(cache) -> (stopped::Bool, retcode::Symbol)`.
Called at `:post_iter` by default.
"""
struct UserStop{F} <: AbstractStoppingCriterion
    f::F
end

function on_event!(c::UserStop, cache, event::Symbol)
    event === :post_iter || return (false, :Default)
    return c.f(cache)
end

# ============================================================================
# Composite — AnyOf
# ============================================================================

"""
    AnyOf(criteria...)

Composite criterion: stops when any constituent stops. Delegates each
event to all sub-criteria in declaration order; the first to fire wins
and its retcode propagates.
"""
struct AnyOf{T<:Tuple} <: AbstractStoppingCriterion
    criteria::T
end

AnyOf(criteria...) = AnyOf(criteria)

function on_event!(c::AnyOf, cache, event::Symbol)
    for crit in c.criteria
        stopped, code = on_event!(crit, cache, event)
        if stopped
            return (true, code)
        end
    end
    return (false, :Default)
end
