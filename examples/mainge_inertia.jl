# mainge_inertia.jl — Custom inertial rule: Nesterov-style schedule.
#
# Pattern demonstrated: the simplest extension contract in the package.
# Inertial rules receive plain scalar arguments — no `ctx`, no `cache`,
# no `state`. Just implement one function returning Float64.
#
# Schedule:  θ_k = (k − 1) / (k + α),  k ≥ 1; θ_0 = 0
#
# This is the Nesterov-style accelerated form used in the convergence
# analyses of inertial Krasnoselskii–Mann algorithms. The parameter `α`
# (default 3.0) must satisfy `α > 2` for the standard convergence proof
# to apply.
#
# Reference: Maingé, P.-É. (2008). Convergence Theorems for Inertial
# KM-Type Algorithms. Journal of Computational and Applied Mathematics
# 219(1): 223–236. https://doi.org/10.1016/j.cam.2007.07.021
#
# Run:
#   cd DFMethods.jl/
#   julia --project=. examples/mainge_inertia.jl

using DFMethods
using LinearAlgebra
using SciMLBase
using CommonSolve

# ── The custom subtype ─────────────────────────────────────────────────
#
# Contract:
#   - subtype `AbstractInertialRule`
#   - implement `inertial_coef(rule, k, xk, xkm1) -> Float64`
#
# Stateless: no `init_state` needed.
struct NesterovInertia <: AbstractInertialRule
    α::Float64
end

NesterovInertia(; α = 3.0) = NesterovInertia(α)

# ── Contract method ────────────────────────────────────────────────────
function DFMethods.inertial_coef(rule::NesterovInertia, k::Int,
                                 xk::AbstractVector, xkm1::AbstractVector)
    k == 0 && return 0.0   # no inertia at k = 0 (no x_{-1})
    return (k - 1) / (k + rule.α)
end

# ── Demo: Dai P2 ───────────────────────────────────────────────────────
F(u, p) = u .- sin.(u)
prob    = NonlinearProblem(F, ones(100))
alg     = DFProjection(; inertial = NesterovInertia(α = 3.0),
                         abstol   = 1e-6,
                         maxiters = 1000)

sol = solve(prob, alg)
println("NesterovInertia(α = 3.0) on Dai P2 (n = 100):")
println("  retcode = ", sol.retcode)
println("  ‖u‖     = ", round(norm(sol.u); digits = 10))
println("  iters   = ", sol.stats.nsteps)
println("  F evals = ", sol.stats.nf)
