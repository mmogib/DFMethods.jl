# extending.jl — How to define your own search direction and line search.
#
# Run:
#   cd DFMethods.jl/
#   julia --project=. examples/extending.jl
#
# Two interfaces, two methods. The rest of the algorithm — inertia,
# projection, hyperplane construction, NonlinearSolve.jl integration —
# is free.

using DFMethods
using LinearAlgebra
using SciMLBase

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
# Contract:
#   - subtype  `AbstractDFLineSearch` with fields `σ::Float64`, `ρ::Float64`
#     (the unified line-search driver reads them)
#   - implement `gamma_k(rule, F_z) -> Float64`
#
# γ_k = ‖F(z_k)‖^p
#   p = 1.0 → equivalent to LSII
#   p = 0.0 → equivalent to LSI (constant)
#   p ∈ (0, 1) → interpolates; empirically gentler on stiff problems

Base.@kwdef struct LSPower <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
    p::Float64 = 0.5
end

DFMethods.gamma_k(rule::LSPower, F_z) = norm(F_z)^rule.p

# ============================================================================
# 3. Compose and solve
# ============================================================================

F(u, p) = u .- sin.(u)               # Dai P2 — solution x* = 0
prob = NonlinearProblem(F, ones(100))

alg_custom = DFProjection(;
    direction  = MPRPL_Direction(),
    linesearch = LSPower(p = 0.5),
    inertial   = Inertial(0.25),
    set        = RealSpace(),
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
# 4. Compare with the paper's default (SpectralThreeTerm + LSII)
# ============================================================================

alg_default = DFProjection(;
    direction  = SpectralThreeTerm(),
    linesearch = LSII(),
    inertial   = Inertial(0.25),
    set        = RealSpace(),
    abstol     = 1e-6,
    maxiters   = 1000,
)

sol_default = solve(prob, alg_default)

println()
println("SpectralThreeTerm + LSII (paper default):")
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
# SpectralThreeTerm and MPRPL both satisfy the direction conditions (proved in their
# respective papers). LSPower with p > 0 satisfies γ_k > 0 whenever
# F(z_k) ≠ 0 — fine for convergence to a non-degenerate solution.
#
# If you implement a direction or line search that violates these
# conditions, the algorithm may still run in practice but you forfeit
# the convergence guarantee.
