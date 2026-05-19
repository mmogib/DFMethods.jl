# extending.jl — How to define your own search direction and line search.
#
# Run:
#   cd DFMethods.jl/
#   julia --project=. examples/extending.jl
#
# Two interfaces, two methods. The rest of the algorithm — inertia,
# iterate update (hyperplane projection, direct projection, or Halpern),
# NonlinearSolve.jl integration — is free.

using DFMethods
using LinearAlgebra
using SciMLBase
using CommonSolve
using LineSearch

# ============================================================================
# 1. Custom search direction: MPRPL (Dai, Chen, Wen, AMC 2015, eq. 4.1)
# ============================================================================
#
# Contract:
#   - subtype  `AbstractSearchDirection`
#   - implement `direction!(d, rule, ctx)` where `ctx` is a NamedTuple with
#     fields Fw, Fw_prev, w, w_prev, d_prev, k, α_prev (see docstring).
#
# The convergence theory of Ibrahim 2026 (Thm 3.1) requires the direction
# to satisfy sufficient descent (eq. 3) and boundedness (eq. 4). MPRPL
# satisfies both — see Dai 2015 §3.
#
# Formula (k ≥ 1):
#   y_{k-1}   = F(w_k) - F(w_{k-1})
#   β_k^PRP  = F(w_k)' y_{k-1} / ‖F(w_{k-1})‖²
#   d_k      = -(1 + β_k^PRP · F(w_k)' d_{k-1} / ‖F(w_k)‖²) F(w_k)
#              + β_k^PRP · d_{k-1}

struct MPRPL_Direction <: AbstractSearchDirection end

function DFMethods.direction!(d, ::MPRPL_Direction, ctx)
    Fw, Fw_prev, d_prev, k = ctx.Fw, ctx.Fw_prev, ctx.d_prev, ctx.k

    if k == 0
        @. d = -Fw
        return d
    end

    # Single pass over the four dot products we need.
    Fw_y   = 0.0   # F(w_k)' y_{k-1}              (numerator of β_PRP)
    Fwm_sq = 0.0   # ‖F(w_{k-1})‖²                (denominator of β_PRP)
    Fw_d   = 0.0   # F(w_k)' d_{k-1}              (for the correction term)
    Fw_sq  = 0.0   # ‖F(w_k)‖²
    @inbounds for i in eachindex(Fw)
        yi      = Fw[i] - Fw_prev[i]
        Fw_y   += Fw[i] * yi
        Fwm_sq += Fw_prev[i]^2
        Fw_d   += Fw[i] * d_prev[i]
        Fw_sq  += Fw[i]^2
    end

    β_PRP   = Fw_y / max(Fwm_sq, eps())
    coeff_F = -(1.0 + β_PRP * Fw_d / max(Fw_sq, eps()))

    @inbounds @simd for i in eachindex(Fw)
        d[i] = coeff_F * Fw[i] + β_PRP * d_prev[i]
    end
    return d
end

# ============================================================================
# 2. Custom line search: fractional power γ_k = ‖F(z)‖^p
# ============================================================================
#
# Contract (v0.2): line searches subtype `LineSearch.AbstractLineSearchAlgorithm`
# and implement two `CommonSolve` methods:
#   - `init(prob, alg, fu, u)` returns a cache `<: LineSearch.AbstractLineSearchCache`
#   - `solve!(cache, u, du)` performs the search and returns a
#     `LineSearch.LineSearchSolution`.
#
# γ_k = ‖F(z_k)‖^p
#   p = 1.0 → equivalent to `ResidualNormBacktrack` (the v0.2 default)
#   p = 0.0 → equivalent to `ConstantBacktrack`
#   p ∈ (0, 1) → interpolates; empirically gentler on stiff problems

Base.@kwdef struct LSPower <: LineSearch.AbstractLineSearchAlgorithm
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
    p::Float64 = 0.5
    maxbt::Int = 50
end

mutable struct LSPowerCache{F} <: LineSearch.AbstractLineSearchCache
    F::F
    alg::LSPower
    z_cache::Vector{Float64}
    fu_cache::Vector{Float64}
    n_evals::Int
end

# Wrap the user's F into a common (out, x) -> nothing form so the inner
# loop can call it uniformly regardless of in-place vs out-of-place.
function _wrap_F_lspower(prob::SciMLBase.NonlinearProblem)
    f, p = prob.f, prob.p
    if SciMLBase.isinplace(f)
        return (out, x) -> (f(out, x, p); nothing)
    else
        return (out, x) -> (val = f(x, p); copyto!(out, val); nothing)
    end
end

function CommonSolve.init(prob::SciMLBase.NonlinearProblem,
                          alg::LSPower, fu, u; kwargs...)
    n = length(u)
    return LSPowerCache(_wrap_F_lspower(prob), alg,
                        Vector{Float64}(undef, n),
                        Vector{Float64}(undef, n), 0)
end

function CommonSolve.solve!(cache::LSPowerCache,
                            u::AbstractVector, du::AbstractVector)
    σ, ρ, p, maxbt = cache.alg.σ, cache.alg.ρ, cache.alg.p, cache.alg.maxbt
    cache.n_evals = 0

    d_norm_sq = 0.0
    @inbounds for j in eachindex(du)
        d_norm_sq += du[j] * du[j]
    end

    α = 1.0
    for _ in 0:maxbt
        @inbounds @simd for j in eachindex(u)
            cache.z_cache[j] = u[j] + α * du[j]
        end
        cache.F(cache.fu_cache, cache.z_cache)
        cache.n_evals += 1

        lhs = 0.0
        @inbounds for j in eachindex(du)
            lhs -= cache.fu_cache[j] * du[j]
        end

        γ   = sqrt(sum(abs2, cache.fu_cache))^p     # γ_k = ‖F(z)‖^p
        rhs = σ * α * γ * d_norm_sq

        if lhs >= rhs
            return LineSearch.LineSearchSolution(α, SciMLBase.ReturnCode.Success)
        end
        α *= ρ
    end
    return LineSearch.LineSearchSolution(α, SciMLBase.ReturnCode.Failure)
end

# ============================================================================
# 3. Compose and solve
# ============================================================================

F(u, p) = u .- sin.(u)               # Dai P2 — solution x* = 0
prob = NonlinearProblem(F, ones(100))     # constraint set lives on the problem;
                                          # an unconstrained NonlinearProblem
                                          # solves on ℝⁿ.

alg_custom = DFProjection(;
    direction  = MPRPL_Direction(),
    linesearch = LSPower(p = 0.5),
    inertial   = Inertial(0.25),
    abstol     = 1e-6,
    maxiters   = 1000,
)

sol_custom = solve(prob, alg_custom)

println("MPRPL_Direction + LSPower(p=0.5):")
println("  retcode  = $(sol_custom.retcode)")
println("  ‖u‖      = $(round(norm(sol_custom.u);  digits=10))")
println("  iters    = $(sol_custom.stats.nsteps)")
println("  F evals  = $(sol_custom.stats.nf)")

# ============================================================================
# 4. Compare with the default (SpectralThreeTerm + ResidualNormBacktrack)
# ============================================================================

alg_default = DFProjection(;
    direction  = SpectralThreeTerm(),
    linesearch = ResidualNormBacktrack(),
    inertial   = Inertial(0.25),
    abstol     = 1e-6,
    maxiters   = 1000,
)

sol_default = solve(prob, alg_default)

println()
println("SpectralThreeTerm + ResidualNormBacktrack (default):")
println("  retcode  = $(sol_default.retcode)")
println("  ‖u‖      = $(round(norm(sol_default.u); digits=10))")
println("  iters    = $(sol_default.stats.nsteps)")
println("  F evals  = $(sol_default.stats.nf)")

# ============================================================================
# Theoretical caveat
# ============================================================================
#
# The framework's convergence theorem (Ibrahim 2026 Thm 3.1) holds when:
#   - the direction satisfies eqs. (3)–(4) (sufficient descent + bounded);
#   - the line search γ_k is bounded below by a positive constant on
#     bounded sets and fits the unified form eq. (7).
#
# SpectralThreeTerm and MPRPL both satisfy the direction conditions (proved in
# their respective papers). LSPower with p > 0 satisfies γ_k > 0 whenever
# F(z_k) ≠ 0 — fine for convergence to a non-degenerate solution.
#
# If you implement a direction or line search that violates these
# conditions, the algorithm may still run in practice but you forfeit
# the convergence guarantee.
