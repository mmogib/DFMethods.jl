# iterate_updates.jl — Pluggable iterate-update strategies (Section A).
#
# Steps 6–7 of the algorithm (hyperplane construction + projection in the
# default case) factor out into an `AbstractIterateUpdate` strategy. Each
# strategy implements
#
#     update_iterate!(x_new, rule, ctx)
#
# where `ctx` is a NamedTuple carrying the inputs needed to produce
# `x_{k+1}` from the post-line-search state `(w, d, α, z, Fw, Fz)`.
#
# Three concrete strategies ship:
#   SolodovSvaiterProjection  — default; hyperplane projection (Solodov & Svaiter 1999)
#   DirectUpdate              — x_{k+1} = project(z, set); simplest baseline
#   HalpernUpdate(β)          — x_{k+1} = β·x_0 + (1-β)·z; averaging variant (Halpern 1967)

# Default: strategies are stateless (see init_state contract in types.jl).
init_state(::AbstractIterateUpdate, prob, x0, alg) = nothing

# ============================================================================
# SolodovSvaiterProjection — hyperplane projection (Solodov & Svaiter 1999)
# ============================================================================

"""
    SolodovSvaiterProjection()

Default iterate-update strategy. Implements the hyperplane projection
scheme of Solodov & Svaiter (1999): given the trial point `z` with
residual `F(z)`, construct the separating hyperplane
`H_k = {x : F(z)' (x − z) ≤ 0}` and project the target
`w − λ_k F(z)` onto `X ∩ H_k` to tolerance
`ε_k = (ζ²/2) ‖λ_k F(z)‖²`.
"""
struct SolodovSvaiterProjection <: AbstractIterateUpdate end

"""
    SolodovSvaiterState

Per-solve state for `SolodovSvaiterProjection`. Holds the projection
target buffer and Dykstra's four inner buffers.
"""
struct SolodovSvaiterState
    proj_target::Vector{Float64}
    proj_p::Vector{Float64}
    proj_q::Vector{Float64}
    proj_scratch::Vector{Float64}
    proj_out_prev::Vector{Float64}
end

function SolodovSvaiterState(n::Int)
    SolodovSvaiterState(
        Vector{Float64}(undef, n),
        zeros(n), zeros(n),
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n),
    )
end

init_state(::SolodovSvaiterProjection, prob, x0, alg) =
    SolodovSvaiterState(length(x0))

function update_iterate!(x_new::AbstractVector,
                          ::SolodovSvaiterProjection, ctx)
    w, d, z      = ctx.w, ctx.d, ctx.z
    Fz           = ctx.Fz
    set          = ctx.set
    ζ            = ctx.ζ
    inner_maxiter = ctx.inner_maxiter
    state        = ctx.state

    # Compute λ_k = F(z)' (w − z) / ‖F(z)‖² and ‖F(z)‖²
    inner_wz   = 0.0
    inner_Fz_z = 0.0
    Fz_norm_sq = 0.0
    @inbounds for i in eachindex(w)
        inner_wz   += Fz[i] * (w[i] - z[i])
        inner_Fz_z += Fz[i] * z[i]
        Fz_norm_sq += Fz[i] * Fz[i]
    end
    λ = inner_wz / Fz_norm_sq

    # target = w - λ·F(z)
    @inbounds @simd for i in eachindex(w)
        state.proj_target[i] = w[i] - λ * Fz[i]
    end

    # ε_k = (ζ²/2) ‖λ F(z)‖²
    ε_k = 0.5 * ζ^2 * λ * λ * Fz_norm_sq

    # Project onto X ∩ H_k
    approx_project_X_halfspace!(x_new, state.proj_target,
                                 set, Fz, inner_Fz_z, ε_k,
                                 state.proj_p, state.proj_q,
                                 state.proj_scratch, state.proj_out_prev;
                                 maxiter = inner_maxiter)
    return x_new
end

# ============================================================================
# DirectUpdate (x_{k+1} = project(z, set))
# ============================================================================

"""
    DirectUpdate()

Stateless update: `x_{k+1} = project(z, set)`. Simplest possible scheme —
take the trial point and project it onto the feasible set. Useful as a
baseline for comparison and as a building block in higher-level methods.
"""
struct DirectUpdate <: AbstractIterateUpdate end

# init_state inherits the AbstractIterateUpdate default — `nothing`.

function update_iterate!(x_new::AbstractVector, ::DirectUpdate, ctx)
    project!(x_new, ctx.z, ctx.set)
    return x_new
end

# ============================================================================
# HalpernUpdate (averaging with the initial iterate)
# ============================================================================

"""
    HalpernUpdate(β)

Halpern-style averaging with feasibility maintenance:

```
x_{k+1} = P_X(β · x_0 + (1 − β) · z_k)
```

where `P_X` is projection onto the problem's feasibility set `X`. The
trial point `z_k = w_k + α_k d_k` is generally infeasible, so the final
projection is required to maintain the framework invariant `x_k ∈ X`.

The parameter `β` is either a scalar (constant across iterations) or a
callable `β(k) -> Float64` (per-iteration schedule, e.g.
`k -> 1 / (k + 2)` for the classical schedule with strong convergence
to the nearest fixed point).

# Reference

Halpern, B. (1967). *Fixed Points of Nonexpanding Maps.* Bulletin of the
AMS **73**(6): 957–961.
[doi:10.1090/S0002-9904-1967-11864-0](https://doi.org/10.1090/S0002-9904-1967-11864-0).
"""
struct HalpernUpdate{B} <: AbstractIterateUpdate
    β::B
end

"""
    HalpernState

Per-solve state for `HalpernUpdate`: holds a copy of the initial
feasible iterate `x_0` and a scratch buffer for the pre-projection
combination `β·x_0 + (1-β)·z_k`.
"""
struct HalpernState
    x0::Vector{Float64}
    combination_buf::Vector{Float64}
end

function init_state(rule::HalpernUpdate, prob, x0, alg)
    # x0 is already projected onto the resolved set by init_cache; copy here.
    x0_copy = copy(collect(Float64, x0))
    return HalpernState(x0_copy, Vector{Float64}(undef, length(x0_copy)))
end

@inline _halpern_β(β::Number, k) = Float64(β)
@inline _halpern_β(β,        k) = Float64(β(k))

function update_iterate!(x_new::AbstractVector, rule::HalpernUpdate, ctx)
    β     = _halpern_β(rule.β, ctx.k)
    state = ctx.state
    buf   = state.combination_buf
    @inbounds @simd for i in eachindex(buf)
        buf[i] = β * state.x0[i] + (1.0 - β) * ctx.z[i]
    end
    project!(x_new, buf, ctx.set)
    return x_new
end
