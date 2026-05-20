# l1_ball.jl — Custom constraint set: ℓ₁ ball.
#
# Pattern demonstrated: an `AbstractConstraintSet` with a non-trivial
# projection algorithm. Constraint sets receive only `(y, x, set)` — no
# `ctx`, no `cache`, no `state` — so the algorithmic content lives
# entirely inside `project!`.
#
# The ℓ₁ ball is  X = { y ∈ Rⁿ : ‖y‖₁ ≤ r }. Projection requires the
# sorted-soft-threshold algorithm: sort |x| descending, locate the
# active set via a cumulative-sum sweep, and soft-threshold.
#
# Reference: Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T.
# (2008). Efficient Projections onto the ℓ₁-Ball for Learning in High
# Dimensions. Proc. 25th ICML, 272–279.
# https://doi.org/10.1145/1390156.1390191
#
# Run:
#   cd DFMethods.jl/
#   julia --project=. examples/l1_ball.jl

using DFMethods
using LinearAlgebra
using SciMLBase
using CommonSolve

# ── The custom subtype ─────────────────────────────────────────────────
#
# Contract:
#   - subtype `AbstractConstraintSet`
#   - implement `project!(y, x, set) -> y` (in-place projection)
#
# Stateless: no `init_state` needed.
struct L1Ball <: AbstractConstraintSet
    r::Float64
    function L1Ball(r::Real)
        r > 0 || throw(ArgumentError("L1Ball: radius must be positive"))
        return new(Float64(r))
    end
end

# ── Contract method ────────────────────────────────────────────────────
#
# Algorithm (Duchi 2008):
#   1. If ‖x‖₁ ≤ r, return x.
#   2. Sort μ = |x| in decreasing order.
#   3. Sweep j = 1..n; track cumulative sum and threshold candidate
#      θ_j = (Σ_{i≤j} μ_i − r) / j.
#      Active count ρ = largest j with μ_j − θ_j > 0; final θ = θ_ρ.
#   4. y_i = sign(x_i) · max(|x_i| − θ, 0).
function DFMethods.project!(y::AbstractVector, x::AbstractVector, set::L1Ball)
    r = set.r

    # Fast path: already feasible.
    s = 0.0
    @inbounds for i in eachindex(x)
        s += abs(x[i])
    end
    if s <= r
        copyto!(y, x)
        return y
    end

    μ = sort!(abs.(x); rev = true)

    cumsum_μ = 0.0
    θ        = 0.0
    @inbounds for j in eachindex(μ)
        cumsum_μ      += μ[j]
        candidate_θ    = (cumsum_μ - r) / j
        μ[j] - candidate_θ > 0 && (θ = candidate_θ)
    end

    @inbounds for i in eachindex(x)
        y[i] = sign(x[i]) * max(abs(x[i]) - θ, 0.0)
    end
    return y
end

# ── Sanity check: project a known vector ───────────────────────────────
let x = [3.0, -2.0, 1.0, 0.5]
    y = similar(x)
    project!(y, x, L1Ball(3.0))
    println("project!([3, -2, 1, 0.5], L1Ball(3)) = ", round.(y; digits = 4))
    println("  ‖y‖₁ = ", round(sum(abs, y); digits = 4), "  (should be ≤ 3)")
    println()
end

# ── Demo: Dai P2 with an ℓ₁-ball constraint ────────────────────────────
#
# The unconstrained solution is u* = 0 (which trivially satisfies the
# constraint). The constraint is enforced throughout iteration via
# the projection, exercising the full contract.
F(u, p)    = u .- sin.(u)
n          = 100
prob_inner = NonlinearProblem(F, 0.3 * ones(n))
prob       = ConstrainedNonlinearProblem(prob_inner, L1Ball(50.0))

alg = DFProjection(; abstol = 1e-6, maxiters = 1000)
sol = solve(prob, alg)
println("DFProjection with L1Ball(50.0) on Dai P2 (n = 100):")
println("  retcode = ", sol.retcode)
println("  ‖u‖₁    = ", round(sum(abs, sol.u); digits = 4))
println("  ‖u‖     = ", round(norm(sol.u); digits = 10))
println("  iters   = ", sol.stats.nsteps)
println("  F evals = ", sol.stats.nf)
