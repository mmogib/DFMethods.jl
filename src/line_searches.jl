# line_searches.jl — Backtracking line searches (LineSearch.jl-aligned).
#
# Three concrete subtypes of `LineSearch.AbstractLineSearchAlgorithm`:
#
#   ConstantBacktrack         γ = 1                          (Armijo classic)
#   ResidualNormBacktrack     γ = ‖F(z)‖                     (default; was LSII)
#   AdaptiveClampedBacktrack  γ = clamp(‖F(z)‖, lo, lo + Δ)  (was LSVII)
#
# Each pairs with a matching `<:LineSearch.AbstractLineSearchCache` that
# exposes `z_cache`, `fu_cache`, and `n_evals` so the DFProjection outer
# loop can avoid a redundant F-evaluation per outer iteration.
#
# Descent condition (Armijo-style):
#
#     -F(z)' du  ≥  σ · α · γ_k · ‖du‖²,    z = u + α·du
#
# The backtracking factor `ρ` shrinks α by ρ until the descent test passes
# or `maxbt` is exceeded. The result is a `LineSearch.LineSearchSolution`
# carrying `step_size` and `retcode`.

# ─── Algorithm structs ──────────────────────────────────────────────────────

"""
    ConstantBacktrack(; σ=0.01, ρ=0.6, maxbt=50)

Backtracking line search with `γ_k ≡ 1` — Armijo's classical form.
"""
Base.@kwdef struct ConstantBacktrack <: LineSearch.AbstractLineSearchAlgorithm
    σ::Float64     = 0.01
    ρ::Float64     = 0.6
    maxbt::Int     = 50
end

"""
    ResidualNormBacktrack(; σ=0.01, ρ=0.6, maxbt=50)

Backtracking with `γ_k = ‖F(z_k)‖`. Default line search for `DFProjection()`.
"""
Base.@kwdef struct ResidualNormBacktrack <: LineSearch.AbstractLineSearchAlgorithm
    σ::Float64     = 0.01
    ρ::Float64     = 0.6
    maxbt::Int     = 50
end

"""
    AdaptiveClampedBacktrack(; σ=0.01, ρ=0.6, lo=1e-4, Δ_init=10.0, maxbt=50)

Backtracking with `γ_k = clamp(‖F(z_k)‖, lo, lo + Δ)`. The `Δ` upper-bound
range is held on the cache (mutable; could adapt across solves in a future
extension; not currently updated).
"""
Base.@kwdef struct AdaptiveClampedBacktrack <: LineSearch.AbstractLineSearchAlgorithm
    σ::Float64     = 0.01
    ρ::Float64     = 0.6
    lo::Float64    = 1e-4
    Δ_init::Float64 = 10.0
    maxbt::Int     = 50
end

# ─── Cache structs ──────────────────────────────────────────────────────────
# Common shape: callable F closure, algorithm config, scratch vectors, and
# the n_evals counter that DFProjection reads.

mutable struct ConstantBacktrackCache{T<:AbstractFloat, F, Alg<:ConstantBacktrack} <: LineSearch.AbstractLineSearchCache
    F::F
    alg::Alg
    z_cache::Vector{T}
    fu_cache::Vector{T}
    n_evals::Int
end

mutable struct ResidualNormBacktrackCache{T<:AbstractFloat, F, Alg<:ResidualNormBacktrack} <: LineSearch.AbstractLineSearchCache
    F::F
    alg::Alg
    z_cache::Vector{T}
    fu_cache::Vector{T}
    n_evals::Int
end

mutable struct AdaptiveClampedBacktrackCache{T<:AbstractFloat, F, Alg<:AdaptiveClampedBacktrack} <: LineSearch.AbstractLineSearchCache
    F::F
    alg::Alg
    z_cache::Vector{T}
    fu_cache::Vector{T}
    n_evals::Int
    Δ::T
end

const _AnyDFBacktrackCache = Union{ConstantBacktrackCache,
                                    ResidualNormBacktrackCache,
                                    AdaptiveClampedBacktrackCache}

# ─── γ_k formulas (dispatched on cache type) ────────────────────────────────

_γ_k(::ConstantBacktrackCache,     fu) = one(eltype(fu))
_γ_k(::ResidualNormBacktrackCache, fu) = (s = zero(eltype(fu)); @inbounds for x in fu; s += x*x end; sqrt(s))
function _γ_k(c::AdaptiveClampedBacktrackCache, fu)
    T = eltype(fu)
    s = zero(T)
    @inbounds for x in fu; s += x*x end
    # c.alg.lo is Float64 (Approach A hyperparam); coerce to T.
    lo_T = T(c.alg.lo)
    return clamp(sqrt(s), lo_T, lo_T + c.Δ)
end

# ─── F adapter: handle in-place vs out-of-place NonlinearProblem.f ──────────

function _wrap_F_into(prob::SciMLBase.NonlinearProblem)
    f, p = prob.f, prob.p
    if SciMLBase.isinplace(f)
        return (out, x) -> (f(out, x, p); nothing)
    else
        return (out, x) -> (val = f(x, p); copyto!(out, val); nothing)
    end
end

# ─── CommonSolve.init ───────────────────────────────────────────────────────

function CommonSolve.init(prob::SciMLBase.NonlinearProblem,
                          alg::ConstantBacktrack, fu, u;
                          stats=nothing, kwargs...)
    F = _wrap_F_into(prob)
    n = length(u)
    T = eltype(u)
    return ConstantBacktrackCache(F, alg,
                                   Vector{T}(undef, n),
                                   Vector{T}(undef, n),
                                   0)
end

function CommonSolve.init(prob::SciMLBase.NonlinearProblem,
                          alg::ResidualNormBacktrack, fu, u;
                          stats=nothing, kwargs...)
    F = _wrap_F_into(prob)
    n = length(u)
    T = eltype(u)
    return ResidualNormBacktrackCache(F, alg,
                                       Vector{T}(undef, n),
                                       Vector{T}(undef, n),
                                       0)
end

function CommonSolve.init(prob::SciMLBase.NonlinearProblem,
                          alg::AdaptiveClampedBacktrack, fu, u;
                          stats=nothing, kwargs...)
    F = _wrap_F_into(prob)
    n = length(u)
    T = eltype(u)
    return AdaptiveClampedBacktrackCache(F, alg,
                                          Vector{T}(undef, n),
                                          Vector{T}(undef, n),
                                          0, T(alg.Δ_init))
end

# ─── Shared backtracking driver ─────────────────────────────────────────────

function _backtrack!(cache::_AnyDFBacktrackCache, u::AbstractVector, du::AbstractVector)
    T = eltype(u)
    # σ, ρ are Float64 hyperparameters (Approach A); coerce to T for
    # type-stable inner loops.
    σ     = T(cache.alg.σ)
    ρ     = T(cache.alg.ρ)
    maxbt = cache.alg.maxbt
    cache.n_evals = 0

    d_norm_sq = zero(T)
    @inbounds for j in eachindex(du)
        d_norm_sq += du[j] * du[j]
    end

    α = one(T)
    for _ in 0:maxbt
        @inbounds @simd for j in eachindex(u)
            cache.z_cache[j] = u[j] + α * du[j]
        end
        cache.F(cache.fu_cache, cache.z_cache)
        cache.n_evals += 1

        lhs = zero(T)
        @inbounds for j in eachindex(du)
            lhs -= cache.fu_cache[j] * du[j]
        end

        γ   = _γ_k(cache, cache.fu_cache)
        rhs = σ * α * γ * d_norm_sq

        if lhs >= rhs
            return LineSearch.LineSearchSolution(α, SciMLBase.ReturnCode.Success)
        end
        α *= ρ
    end
    return LineSearch.LineSearchSolution(α, SciMLBase.ReturnCode.Failure)
end

# ─── CommonSolve.solve! ─────────────────────────────────────────────────────

CommonSolve.solve!(cache::ConstantBacktrackCache,        u, du) = _backtrack!(cache, u, du)
CommonSolve.solve!(cache::ResidualNormBacktrackCache,    u, du) = _backtrack!(cache, u, du)
CommonSolve.solve!(cache::AdaptiveClampedBacktrackCache, u, du) = _backtrack!(cache, u, du)
