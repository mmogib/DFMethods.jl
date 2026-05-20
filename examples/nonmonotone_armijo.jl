# nonmonotone_armijo.jl — Custom line search: non-monotone Armijo.
#
# Pattern demonstrated: the line-search contract via `CommonSolve.init` /
# `CommonSolve.solve!`, with a mutable cache that carries state ACROSS
# `solve!` calls (the residual-norm history window). All other extension
# contracts in DFMethods are stateless or use `init_state`; line searches
# are unique in that the cache itself is the state container.
#
# Reference: Grippo, L., Lampariello, F., & Lucidi, S. (1986). A
# nonmonotone line search technique for Newton's method. SIAM Journal on
# Numerical Analysis 23(4): 707–716.
# https://doi.org/10.1137/0723046
#
# Acceptance rule: accept step α if
#     ‖F(z)‖² ≤ max_{j ∈ window} ‖F(z_j)‖² − σ α² ‖d‖²
# where the window is the M most recent accepted residuals. Standard
# (monotone) Armijo is recovered with M = 1.
#
# Run:
#   cd DFMethods.jl/
#   julia --project=. examples/nonmonotone_armijo.jl

using DFMethods
using LinearAlgebra
using SciMLBase
using CommonSolve
using LineSearch

# ── The custom subtype ─────────────────────────────────────────────────
Base.@kwdef struct NonmonotoneArmijo <: LineSearch.AbstractLineSearchAlgorithm
    σ::Float64    = 1e-4
    ρ::Float64    = 0.6
    memory::Int   = 5       # M, sliding-window size
    maxbt::Int    = 50
end

# ── Cache: mutable, holds the residual history across solve! calls ────
mutable struct NonmonotoneCache{F} <: LineSearch.AbstractLineSearchCache
    F::F
    alg::NonmonotoneArmijo
    z_cache::Vector{Float64}
    fu_cache::Vector{Float64}
    history::Vector{Float64}   # recent ‖F(z)‖² values (oldest first)
    n_evals::Int
end

# Adapter to call the user's F uniformly (handles both in-place and out-of-place).
function _wrap_F(prob::SciMLBase.NonlinearProblem)
    f, p = prob.f, prob.p
    if SciMLBase.isinplace(f)
        return (out, x) -> (f(out, x, p); nothing)
    else
        return (out, x) -> (val = f(x, p); copyto!(out, val); nothing)
    end
end

# ── `init` returns the cache. Called once per solve, at init_cache time. ──
function CommonSolve.init(prob::SciMLBase.NonlinearProblem,
                          alg::NonmonotoneArmijo, fu, u; kwargs...)
    return NonmonotoneCache(_wrap_F(prob), alg,
                            Vector{Float64}(undef, length(u)),
                            Vector{Float64}(undef, length(u)),
                            Float64[],   # empty history at solve start
                            0)
end

# ── `solve!` performs the backtracking search; updates history on accept. ──
function CommonSolve.solve!(cache::NonmonotoneCache,
                            u::AbstractVector, du::AbstractVector)
    σ, ρ, M, maxbt = cache.alg.σ, cache.alg.ρ, cache.alg.memory, cache.alg.maxbt
    cache.n_evals = 0

    d_norm_sq = 0.0
    @inbounds for j in eachindex(du)
        d_norm_sq += du[j] * du[j]
    end

    # Non-monotone reference: max of recent ‖F‖²; +Inf on first iteration
    # accepts any step that produces a finite residual.
    Fmax_sq = isempty(cache.history) ? Inf : maximum(cache.history)

    α = 1.0
    for _ in 0:maxbt
        @inbounds @simd for j in eachindex(u)
            cache.z_cache[j] = u[j] + α * du[j]
        end
        cache.F(cache.fu_cache, cache.z_cache)
        cache.n_evals += 1

        Fz_sq = sum(abs2, cache.fu_cache)

        if Fz_sq <= Fmax_sq - σ * α^2 * d_norm_sq
            push!(cache.history, Fz_sq)
            length(cache.history) > M && popfirst!(cache.history)
            return LineSearch.LineSearchSolution(α, SciMLBase.ReturnCode.Success)
        end
        α *= ρ
    end
    return LineSearch.LineSearchSolution(α, SciMLBase.ReturnCode.Failure)
end

# ── Demo: Dai P2 ───────────────────────────────────────────────────────
F(u, p) = u .- sin.(u)
prob    = NonlinearProblem(F, ones(100))
alg     = DFProjection(; linesearch = NonmonotoneArmijo(memory = 5),
                         abstol     = 1e-6,
                         maxiters   = 1000)

sol = solve(prob, alg)
println("NonmonotoneArmijo line search on Dai P2 (n = 100):")
println("  retcode = ", sol.retcode)
println("  ‖u‖     = ", round(norm(sol.u); digits = 10))
println("  iters   = ", sol.stats.nsteps)
println("  F evals = ", sol.stats.nf)
