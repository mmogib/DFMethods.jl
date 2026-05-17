# line_searches.jl — Derivative-free line searches (LS I–LS VII).
#
# Ibrahim 2026 eq. (7) unifies seven backtracking variants. Each variant
# defines a per-iteration multiplier γ_k such that the smallest i ≥ 0
# with α_k = ρ^i is accepted when
#
#     -F(w_k + α_k d_k)' d_k  ≥  σ · α_k · γ_k · ‖d_k‖²
#
# γ_k depends on F at the trial point z = w_k + α_k d_k.
#
# Notation: the paper calls the backtracking factor `γ`; we use `ρ` here
# (standard optimization notation) to avoid colliding with `γ_k`.

"""
    AbstractDFLineSearch

Supertype for derivative-free line-search rules. Each subtype defines
the multiplier `gamma_k(rule, F_z)` entering the unified descent
condition.

All concrete rules carry the scalars `σ` (Armijo coefficient) and `ρ`
(backtracking factor), with defaults `σ = 0.01`, `ρ = 0.6` (paper).
"""
abstract type AbstractDFLineSearch end

"""
    gamma_k(rule::AbstractDFLineSearch, F_z::AbstractVector) -> Float64

Per-iteration line-search multiplier ``\\gamma_k`` entering the unified
descent condition (Ibrahim 2026 eq. 7)

```math
-\\psi(w_k + \\alpha_k d_k)^\\top d_k \\;\\geq\\; \\sigma\\, \\alpha_k\\, \\gamma_k\\, \\|d_k\\|^2.
```

Each concrete [`AbstractDFLineSearch`](@ref) subtype (`LSI`–`LSVII`)
supplies its own formula; see the individual line-search docstrings.
"""
function gamma_k end

# ============================================================================
# LSI — γ_k ≡ 1 (plain Armijo-style)
# ============================================================================

"""
    LSI(; σ=0.01, ρ=0.6)

`γ_k = 1`. Plain Armijo on the dual residual.
"""
Base.@kwdef struct LSI <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
end

gamma_k(::LSI, F_z::AbstractVector) = 1.0

# ============================================================================
# LSII — γ_k = ‖F(z_k)‖
# ============================================================================

"""
    LSII(; σ=0.01, ρ=0.6)

`γ_k = ‖F(z_k)‖`. Scales the descent test with the residual at the
trial point.
"""
Base.@kwdef struct LSII <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
end

gamma_k(::LSII, F_z::AbstractVector) = norm(F_z)

# ============================================================================
# LSIII — γ_k = γ_{k1} = ‖F‖ / (1 + ‖F‖)  (bounded in (0, 1))
# ============================================================================

"""
    LSIII(; σ=0.01, ρ=0.6)

`γ_k = ‖F(z_k)‖ / (1 + ‖F(z_k)‖)`. Saturates at 1 for large residuals.
"""
Base.@kwdef struct LSIII <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
end

function gamma_k(::LSIII, F_z::AbstractVector)
    n = norm(F_z)
    return n / (1.0 + n)
end

# ============================================================================
# LSIV — γ_k = γ_{k2} = τ + (1-τ) ‖F‖,  τ ∈ (0, 1]
# ============================================================================

"""
    LSIV(; σ=0.01, ρ=0.6, τ=0.5)

`γ_k = τ + (1-τ) · ‖F(z_k)‖`. Paper allows `τ ∈ [τ_min, τ_max] ⊆ (0, 1]`
to vary across iterations; Phase 1 uses a fixed `τ`.
"""
Base.@kwdef struct LSIV <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
    τ::Float64 = 0.5
end

function gamma_k(rule::LSIV, F_z::AbstractVector)
    return rule.τ + (1.0 - rule.τ) * norm(F_z)
end

# ============================================================================
# LSV — γ_k = γ_{k3} = min(1, ‖F‖)
# ============================================================================

"""
    LSV(; σ=0.01, ρ=0.6)

`γ_k = min(1, ‖F(z_k)‖)`. Caps large residuals at 1.
"""
Base.@kwdef struct LSV <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
end

gamma_k(::LSV, F_z::AbstractVector) = min(1.0, norm(F_z))

# ============================================================================
# LSVI — γ_k = γ_{k4} = clamp(‖F‖, lo, hi),  0 < lo ≪ hi
# ============================================================================

"""
    LSVI(; σ=0.01, ρ=0.6, lo=1e-4, hi=10.0)

`γ_k = clamp(‖F(z_k)‖, lo, hi)`. Paper's projection onto
``[\\bar{α}, α]``. `lo` plays the role of ``\\bar{α}``, `hi` the role of
``α`` in eq. (vi).
"""
Base.@kwdef struct LSVI <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
    lo::Float64 = 1e-4
    hi::Float64 = 10.0
end

function gamma_k(rule::LSVI, F_z::AbstractVector)
    return clamp(norm(F_z), rule.lo, rule.hi)
end

# ============================================================================
# LSVII — adaptive variant of LSVI (paper eq. (vii))
# ============================================================================

"""
    LSVII(; σ=0.01, ρ=0.6, lo=1e-4, Δ_init=1.0)

Adaptive variant of LSVI with shrinking upper bound `Δ`:

- For `k = 1, 2, 3`: behaves like LSVI with `hi = lo + Δ`, then
  `Δ ← min(α_k, Δ)`.
- For `k ≥ 4`: `Δ ← min(α_k, Δ) / 4` before forming `hi = lo + Δ`.

This rule carries state across iterations. Phase 1 exposes the
parameter struct and a `gamma_k(rule, F_z; Δ)` keyword form that the
Phase 2 cache will drive.
"""
Base.@kwdef struct LSVII <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
    lo::Float64 = 1e-4
    Δ_init::Float64 = 1.0
end

function gamma_k(rule::LSVII, F_z::AbstractVector; Δ::Float64 = rule.Δ_init)
    return clamp(norm(F_z), rule.lo, rule.lo + Δ)
end

# ============================================================================
# Driver: backtracking loop
# ============================================================================

"""
    linesearch!(rule, F, w, d, z_buf, Fz_buf; maxbt=50)
        -> (α::Float64, F_z::AbstractVector, n_evals::Int, ok::Bool)

Backtracking driver satisfying eq. (7):

```
-F(w + α d)' d  ≥  σ · α · γ_k · ‖d‖²
```

Tries `α = ρ^i` for `i = 0, 1, …, maxbt`, returning the first `α`
that satisfies the inequality. Returns `ok = false` if no acceptable
step is found within `maxbt` backtracks (caller decides whether to
retry with a different direction or flag a failure).

# Arguments
- `rule`: an `AbstractDFLineSearch`.
- `F`: function `F(x) -> Vector`. Out-of-place for Phase 1; Phase 3
  will route through `NonlinearProblem` and support in-place.
- `w`, `d`: current iterate `w_k` and direction `d_k`.
- `z_buf`, `Fz_buf`: pre-allocated buffers for `w + α d` and `F(z)`.

# Returns
- `α`: accepted step size.
- `F_z`: `F` at the accepted trial point (`Fz_buf` after the final call).
- `n_evals`: number of `F` evaluations (one per backtrack attempt).
- `ok`: whether the inequality was satisfied within `maxbt` backtracks.
"""
function linesearch!(rule::AbstractDFLineSearch,
                     F,
                     w::AbstractVector,
                     d::AbstractVector,
                     z_buf::AbstractVector,
                     Fz_buf::AbstractVector;
                     maxbt::Int = 50)
    σ = rule.σ
    ρ = rule.ρ

    d_norm_sq = 0.0
    @inbounds for j in eachindex(d)
        d_norm_sq += d[j] * d[j]
    end

    α = 1.0
    n_evals = 0
    for _ in 0:maxbt
        # z = w + α d
        @inbounds @simd for j in eachindex(w)
            z_buf[j] = w[j] + α * d[j]
        end
        # F_z = F(z) — out-of-place
        Fz_value = F(z_buf)
        @inbounds @simd for j in eachindex(Fz_buf)
            Fz_buf[j] = Fz_value[j]
        end
        n_evals += 1

        # LHS = -F(z)' d
        lhs = 0.0
        @inbounds for j in eachindex(d)
            lhs -= Fz_buf[j] * d[j]
        end

        γ = gamma_k(rule, Fz_buf)
        rhs = σ * α * γ * d_norm_sq

        if lhs >= rhs
            return (α, Fz_buf, n_evals, true)
        end

        α *= ρ
    end
    return (α, Fz_buf, n_evals, false)
end

# ============================================================================
# v0.2 line searches (Section B) — LineSearch.jl-aligned
# ============================================================================
#
# Three concrete line searches that subtype `LineSearch.AbstractLineSearchAlgorithm`
# and implement the standard `CommonSolve.init` / `CommonSolve.solve!` contract.
# Differ only in the γ_k formula entering the Armijo-style descent test
#
#     -F(z)' du  ≥  σ · α · γ_k · ‖du‖²,    z = u + α·du
#
# Each algorithm has a matching cache that exposes `fu_cache` (F at the
# accepted trial point) and `n_evals` so the DFProjection outer loop can
# avoid a redundant F-evaluation per outer iteration.

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

Backtracking with `γ_k = ‖F(z_k)‖`. Default line search for `DFProjection()`
(empirical winner from the s30 benchmark for the SpectralThreeTerm direction).
"""
Base.@kwdef struct ResidualNormBacktrack <: LineSearch.AbstractLineSearchAlgorithm
    σ::Float64     = 0.01
    ρ::Float64     = 0.6
    maxbt::Int     = 50
end

"""
    AdaptiveClampedBacktrack(; σ=0.01, ρ=0.6, lo=1e-4, Δ_init=10.0, maxbt=50)

Backtracking with `γ_k = clamp(‖F(z_k)‖, lo, lo + Δ)`. The `Δ` upper-bound
range adapts across iterations (LSVII analog).
"""
Base.@kwdef struct AdaptiveClampedBacktrack <: LineSearch.AbstractLineSearchAlgorithm
    σ::Float64     = 0.01
    ρ::Float64     = 0.6
    lo::Float64    = 1e-4
    Δ_init::Float64 = 10.0
    maxbt::Int     = 50
end

# ─── Cache types ────────────────────────────────────────────────────────────

# All three caches share the same shape: a callable F closure, the
# algorithm config, two scratch vectors (z_cache, fu_cache), and an eval
# counter that the outer DFProjection loop reads.

mutable struct ConstantBacktrackCache{F, Alg<:ConstantBacktrack} <: LineSearch.AbstractLineSearchCache
    F::F
    alg::Alg
    z_cache::Vector{Float64}
    fu_cache::Vector{Float64}
    n_evals::Int
end

mutable struct ResidualNormBacktrackCache{F, Alg<:ResidualNormBacktrack} <: LineSearch.AbstractLineSearchCache
    F::F
    alg::Alg
    z_cache::Vector{Float64}
    fu_cache::Vector{Float64}
    n_evals::Int
end

mutable struct AdaptiveClampedBacktrackCache{F, Alg<:AdaptiveClampedBacktrack} <: LineSearch.AbstractLineSearchCache
    F::F
    alg::Alg
    z_cache::Vector{Float64}
    fu_cache::Vector{Float64}
    n_evals::Int
    Δ::Float64                              # mutable; could adapt across solves
end

# Union over all DFMethods backtrack caches (for shared `_backtrack!`).
const _AnyDFBacktrackCache = Union{ConstantBacktrackCache,
                                    ResidualNormBacktrackCache,
                                    AdaptiveClampedBacktrackCache}

# ─── γ_k formula per cache ──────────────────────────────────────────────────

_γ_k(::ConstantBacktrackCache,     fu) = 1.0
_γ_k(::ResidualNormBacktrackCache, fu) = (s = 0.0; @inbounds for x in fu; s += x*x end; sqrt(s))
function _γ_k(c::AdaptiveClampedBacktrackCache, fu)
    s = 0.0
    @inbounds for x in fu; s += x*x end
    return clamp(sqrt(s), c.alg.lo, c.alg.lo + c.Δ)
end

# ─── F adapter: handle in-place vs out-of-place NonlinearProblem ────────────

# Wraps the problem's f into a `(out, x) -> nothing` writer that fills `out`.
# For out-of-place f(u, p), broadcasts the returned vector into out.
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
    return ConstantBacktrackCache(F, alg,
                                   Vector{Float64}(undef, n),
                                   Vector{Float64}(undef, n),
                                   0)
end

function CommonSolve.init(prob::SciMLBase.NonlinearProblem,
                          alg::ResidualNormBacktrack, fu, u;
                          stats=nothing, kwargs...)
    F = _wrap_F_into(prob)
    n = length(u)
    return ResidualNormBacktrackCache(F, alg,
                                       Vector{Float64}(undef, n),
                                       Vector{Float64}(undef, n),
                                       0)
end

function CommonSolve.init(prob::SciMLBase.NonlinearProblem,
                          alg::AdaptiveClampedBacktrack, fu, u;
                          stats=nothing, kwargs...)
    F = _wrap_F_into(prob)
    n = length(u)
    return AdaptiveClampedBacktrackCache(F, alg,
                                          Vector{Float64}(undef, n),
                                          Vector{Float64}(undef, n),
                                          0, alg.Δ_init)
end

# ─── Shared backtracking driver ─────────────────────────────────────────────

function _backtrack!(cache::_AnyDFBacktrackCache, u::AbstractVector, du::AbstractVector)
    σ, ρ, maxbt = cache.alg.σ, cache.alg.ρ, cache.alg.maxbt
    cache.n_evals = 0

    # ‖du‖²
    d_norm_sq = 0.0
    @inbounds for j in eachindex(du)
        d_norm_sq += du[j] * du[j]
    end

    α = 1.0
    for _ in 0:maxbt
        # z = u + α · du
        @inbounds @simd for j in eachindex(u)
            cache.z_cache[j] = u[j] + α * du[j]
        end
        # F(z) → fu_cache
        cache.F(cache.fu_cache, cache.z_cache)
        cache.n_evals += 1

        # LHS = -F(z)' du
        lhs = 0.0
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
