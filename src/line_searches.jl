# line_searches.jl — Derivative-free line searches (LS I–LS VII).
#
# Ibrahim 2026 eq. (7) unifies seven backtracking variants. Each variant
# defines a per-iteration multiplier γ_k such that the smallest i ≥ 0
# with α_k = ρ^i is accepted when
#
#     -ψ(w_k + α_k d_k)' d_k  ≥  σ · α_k · γ_k · ‖d_k‖²
#
# γ_k depends on ψ at the trial point z = w_k + α_k d_k.
#
# Notation: the paper calls the backtracking factor `γ`; we use `ρ` here
# (standard optimization notation) to avoid colliding with `γ_k`.

"""
    AbstractDFLineSearch

Supertype for derivative-free line-search rules. Each subtype defines
the multiplier `gamma_k(rule, ψ_z)` entering the unified descent
condition.

All concrete rules carry the scalars `σ` (Armijo coefficient) and `ρ`
(backtracking factor), with defaults `σ = 0.01`, `ρ = 0.6` (paper).
"""
abstract type AbstractDFLineSearch end

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

gamma_k(::LSI, ψ_z::AbstractVector) = 1.0

# ============================================================================
# LSII — γ_k = ‖ψ(z_k)‖
# ============================================================================

"""
    LSII(; σ=0.01, ρ=0.6)

`γ_k = ‖ψ(z_k)‖`. Scales the descent test with the residual at the
trial point.
"""
Base.@kwdef struct LSII <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
end

gamma_k(::LSII, ψ_z::AbstractVector) = norm(ψ_z)

# ============================================================================
# LSIII — γ_k = γ_{k1} = ‖ψ‖ / (1 + ‖ψ‖)  (bounded in (0, 1))
# ============================================================================

"""
    LSIII(; σ=0.01, ρ=0.6)

`γ_k = ‖ψ(z_k)‖ / (1 + ‖ψ(z_k)‖)`. Saturates at 1 for large residuals.
"""
Base.@kwdef struct LSIII <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
end

function gamma_k(::LSIII, ψ_z::AbstractVector)
    n = norm(ψ_z)
    return n / (1.0 + n)
end

# ============================================================================
# LSIV — γ_k = γ_{k2} = τ + (1-τ) ‖ψ‖,  τ ∈ (0, 1]
# ============================================================================

"""
    LSIV(; σ=0.01, ρ=0.6, τ=0.5)

`γ_k = τ + (1-τ) · ‖ψ(z_k)‖`. Paper allows `τ ∈ [τ_min, τ_max] ⊆ (0, 1]`
to vary across iterations; Phase 1 uses a fixed `τ`.
"""
Base.@kwdef struct LSIV <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
    τ::Float64 = 0.5
end

function gamma_k(rule::LSIV, ψ_z::AbstractVector)
    return rule.τ + (1.0 - rule.τ) * norm(ψ_z)
end

# ============================================================================
# LSV — γ_k = γ_{k3} = min(1, ‖ψ‖)
# ============================================================================

"""
    LSV(; σ=0.01, ρ=0.6)

`γ_k = min(1, ‖ψ(z_k)‖)`. Caps large residuals at 1.
"""
Base.@kwdef struct LSV <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
end

gamma_k(::LSV, ψ_z::AbstractVector) = min(1.0, norm(ψ_z))

# ============================================================================
# LSVI — γ_k = γ_{k4} = clamp(‖ψ‖, lo, hi),  0 < lo ≪ hi
# ============================================================================

"""
    LSVI(; σ=0.01, ρ=0.6, lo=1e-4, hi=10.0)

`γ_k = clamp(‖ψ(z_k)‖, lo, hi)`. Paper's projection onto
``[\\bar{α}, α]``. `lo` plays the role of ``\\bar{α}``, `hi` the role of
``α`` in eq. (vi).
"""
Base.@kwdef struct LSVI <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
    lo::Float64 = 1e-4
    hi::Float64 = 10.0
end

function gamma_k(rule::LSVI, ψ_z::AbstractVector)
    return clamp(norm(ψ_z), rule.lo, rule.hi)
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
parameter struct and a `gamma_k(rule, ψ_z; Δ)` keyword form that the
Phase 2 cache will drive.
"""
Base.@kwdef struct LSVII <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
    lo::Float64 = 1e-4
    Δ_init::Float64 = 1.0
end

function gamma_k(rule::LSVII, ψ_z::AbstractVector; Δ::Float64 = rule.Δ_init)
    return clamp(norm(ψ_z), rule.lo, rule.lo + Δ)
end

# ============================================================================
# Driver: backtracking loop
# ============================================================================

"""
    linesearch!(rule, ψ, w, d, z_buf, ψz_buf; maxbt=50)
        -> (α::Float64, ψ_z::AbstractVector, n_evals::Int, ok::Bool)

Backtracking driver satisfying eq. (7):

```
-ψ(w + α d)' d  ≥  σ · α · γ_k · ‖d‖²
```

Tries `α = ρ^i` for `i = 0, 1, …, maxbt`, returning the first `α`
that satisfies the inequality. Returns `ok = false` if no acceptable
step is found within `maxbt` backtracks (caller decides whether to
retry with a different direction or flag a failure).

# Arguments
- `rule`: an `AbstractDFLineSearch`.
- `ψ`: function `ψ(x) -> Vector`. Out-of-place for Phase 1; Phase 3
  will route through `NonlinearProblem` and support in-place.
- `w`, `d`: current iterate `w_k` and direction `d_k`.
- `z_buf`, `ψz_buf`: pre-allocated buffers for `w + α d` and `ψ(z)`.

# Returns
- `α`: accepted step size.
- `ψ_z`: `ψ` at the accepted trial point (`ψz_buf` after the final call).
- `n_evals`: number of `ψ` evaluations (one per backtrack attempt).
- `ok`: whether the inequality was satisfied within `maxbt` backtracks.
"""
function linesearch!(rule::AbstractDFLineSearch,
                     ψ,
                     w::AbstractVector,
                     d::AbstractVector,
                     z_buf::AbstractVector,
                     ψz_buf::AbstractVector;
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
        # ψ_z = ψ(z) — out-of-place
        ψz_value = ψ(z_buf)
        @inbounds @simd for j in eachindex(ψz_buf)
            ψz_buf[j] = ψz_value[j]
        end
        n_evals += 1

        # LHS = -ψ(z)' d
        lhs = 0.0
        @inbounds for j in eachindex(d)
            lhs -= ψz_buf[j] * d[j]
        end

        γ = gamma_k(rule, ψz_buf)
        rhs = σ * α * γ * d_norm_sq

        if lhs >= rhs
            return (α, ψz_buf, n_evals, true)
        end

        α *= ρ
    end
    return (α, ψz_buf, n_evals, false)
end
