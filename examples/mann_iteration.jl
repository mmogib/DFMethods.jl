# mann_iteration.jl — Custom iterate-update strategy: Mann iteration.
#
# Pattern demonstrated: the `init_state` opt-in for per-solve state,
# surfaced to `update_iterate!` via `ctx.state`. This is the canonical
# pattern for a custom update rule that needs information not exposed
# directly through ctx (here: the previous iterate `x_k`).
#
# Mann iteration anchors the next iterate to a convex combination of
# the current iterate and the projected trial point:
#
#     x_{k+1} = α_k · x_k + (1 - α_k) · P_X(z_k)
#
# with α_k ∈ (0, 1). Classical schedule: α_k = 1/(k + 2). Unlike
# `HalpernUpdate`, the anchor moves with the iteration (x_k), not the
# initial point (x_0).
#
# Reference (general lineage):
#   Mann, W. R. (1953). Mean Value Methods in Iteration. Proc. AMS
#   4(3): 506–510. https://doi.org/10.1090/S0002-9939-1953-0054846-3
#
# Run:
#   cd DFMethods.jl/
#   julia --project=. examples/mann_iteration.jl

using DFMethods
using LinearAlgebra
using SciMLBase
using CommonSolve

# ── The custom subtype ─────────────────────────────────────────────────
#
# Contract:
#   - subtype `AbstractIterateUpdate`
#   - implement `update_iterate!(x_new, rule, ctx)`
#   - optionally declare `init_state(rule, prob, x0, alg)` for per-solve scratch.
#
# `α` is either a scalar (constant α_k) or a callable `k -> Float64`
# (schedule). `MannIteration(k -> 1/(k+2))` recovers the classical form.
struct MannIteration{A} <: AbstractIterateUpdate
    α::A
end

# ── Per-solve state ────────────────────────────────────────────────────
#
# `update_iterate!`'s ctx does NOT expose `x_k` directly — only `w`
# (which equals `x_k` only when there is no inertia). So we hold `x_k`
# in state, refreshed on each call. We also keep scratch for `P_X(z_k)`.
mutable struct MannState{T<:AbstractFloat}
    xk::Vector{T}
    proj_z::Vector{T}
end

# The framework calls this once per solve to allocate state. The third
# arg `x` is the projected, T-typed initial iterate from init_cache.
function DFMethods.init_state(::MannIteration, prob, x, alg)
    T = eltype(x)
    return MannState{T}(copy(x), similar(x))
end

# ── Contract method ────────────────────────────────────────────────────
function DFMethods.update_iterate!(x_new::AbstractVector,
                                   rule::MannIteration, ctx)
    state = ctx.state
    T     = eltype(x_new)
    αk    = T(rule.α isa Number ? rule.α : rule.α(ctx.k))

    # Project the trial point z onto the feasibility set X.
    project!(state.proj_z, ctx.z, ctx.set)

    # x_{k+1} = α_k · x_k + (1 − α_k) · P_X(z_k)
    @inbounds @simd for i in eachindex(x_new)
        x_new[i] = αk * state.xk[i] + (one(T) - αk) * state.proj_z[i]
    end

    # Update state for next iteration: state.xk ← x_new.
    copyto!(state.xk, x_new)
    return x_new
end

# ── Demo: Dai P2 ───────────────────────────────────────────────────────
F(u, p) = u .- sin.(u)
prob    = NonlinearProblem(F, ones(100))
alg     = DFProjection(;
    iterate_update = MannIteration(k -> 1.0 / (k + 2)),
    abstol         = 1e-6,
    maxiters       = 1000,
)

sol = solve(prob, alg)
println("MannIteration(k -> 1/(k+2)) on Dai P2 (n = 100):")
println("  retcode = ", sol.retcode)
println("  ‖u‖     = ", round(norm(sol.u); digits = 10))
println("  iters   = ", sol.stats.nsteps)
println("  F evals = ", sol.stats.nf)
