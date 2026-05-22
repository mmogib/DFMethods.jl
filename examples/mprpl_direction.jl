# mprpl_direction.jl — Custom search direction: MPRPL.
#
# Pattern demonstrated: stateless `ctx`-based contract using the history
# fields (`Fw_prev`, `w_prev`, `d_prev`) of `direction!`'s ctx. Difference-
# based conjugate-gradient-style directions follow this shape.
#
# Reference: Dai, Z., Chen, X., & Wen, F. (2015). A Modified Perry's
# Conjugate Gradient Method-Based Derivative-Free Method for Solving
# Large-Scale Nonlinear Monotone Equations. Applied Mathematics and
# Computation 270: 378–386.
# https://doi.org/10.1016/j.amc.2015.08.014
#
# Run:
#   cd DFMethods.jl/
#   julia --project=. examples/mprpl_direction.jl

using DFMethods
using LinearAlgebra
using SciMLBase
using CommonSolve

# ── The custom subtype ─────────────────────────────────────────────────
#
# Contract:
#   - subtype  `AbstractSearchDirection`
#   - implement `direction!(d, rule, ctx)`; see the "Extension contracts"
#     section of the Extending docs page for the full ctx field list.
#
# MPRPL is stateless: no `init_state` needed (the default returns `nothing`).
struct MPRPL <: AbstractSearchDirection end

# ── Contract method ────────────────────────────────────────────────────
#
# Formula (k ≥ 1):
#   y_{k-1}   = F(w_k) - F(w_{k-1})
#   β_k^PRP  = F(w_k)' y_{k-1} / ‖F(w_{k-1})‖²
#   d_k      = -(1 + β_k^PRP · F(w_k)' d_{k-1} / ‖F(w_k)‖²) F(w_k)
#              + β_k^PRP · d_{k-1}
function DFMethods.direction!(d, ::MPRPL, ctx)
    Fw, Fw_prev, d_prev, k = ctx.Fw, ctx.Fw_prev, ctx.d_prev, ctx.k

    # k = 0 fallback: history fields are uninitialized. Use steepest descent.
    if k == 0
        @. d = -Fw
        return d
    end
    T = eltype(Fw)

    Fw_y   = zero(T)   # F(w_k)' y_{k-1}
    Fwm_sq = zero(T)   # ‖F(w_{k-1})‖²
    Fw_d   = zero(T)   # F(w_k)' d_{k-1}
    Fw_sq  = zero(T)   # ‖F(w_k)‖²
    @inbounds for i in eachindex(Fw)
        yi      = Fw[i] - Fw_prev[i]
        Fw_y   += Fw[i] * yi
        Fwm_sq += Fw_prev[i]^2
        Fw_d   += Fw[i] * d_prev[i]
        Fw_sq  += Fw[i]^2
    end

    β_PRP   = Fw_y / max(Fwm_sq, eps(T))
    coeff_F = -(one(T) + β_PRP * Fw_d / max(Fw_sq, eps(T)))

    @inbounds @simd for i in eachindex(Fw)
        d[i] = coeff_F * Fw[i] + β_PRP * d_prev[i]
    end
    return d
end

# ── Demo: Dai P2 — F(u) = u - sin(u), solution u* = 0 ──────────────────
F(u, p) = u .- sin.(u)
prob    = NonlinearProblem(F, ones(100))
alg     = DFProjection(; direction = MPRPL(),
                         abstol    = 1e-6,
                         maxiters  = 1000)

sol = solve(prob, alg)
println("MPRPL direction on Dai P2 (n = 100):")
println("  retcode = ", sol.retcode)
println("  ‖u‖     = ", round(norm(sol.u); digits = 10))
println("  iters   = ", sol.stats.nsteps)
println("  F evals = ", sol.stats.nf)
