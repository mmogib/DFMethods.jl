# inertial.jl — Inertial extrapolation rules.

"""
    AbstractInertialRule

Supertype for inertial-extrapolation rules. A rule produces the inertial
coefficient ``θ_k`` from the iteration index, current iterate, and
previous iterate. The inertial point is then

```
w_k = x_k + θ_k (x_k - x_{k-1}).
```
"""
abstract type AbstractInertialRule end

"""
    inertial_coef(rule, k, xk, xkm1) -> θ_k::Float64

Compute the inertial coefficient. `k` is the iteration index (0-based);
`xk` is the current iterate; `xkm1` is the previous iterate.
"""
function inertial_coef end

# ============================================================================
# NoInertial: θ_k ≡ 0 (w_k = x_k always)
# ============================================================================

"""
    NoInertial()

Disables inertia: ``w_k = x_k``. Useful for ablation comparisons against
the inertial variant.

```jldoctest
julia> inertial_coef(NoInertial(), 5, [1.0, 2.0], [0.0, 1.0])
0.0
```
"""
struct NoInertial <: AbstractInertialRule end

inertial_coef(::NoInertial, k, xk, xkm1) = 0.0

# ============================================================================
# Inertial: summable-step extrapolation rule
# ============================================================================

"""
    Inertial(θ = 0.25)

Standard summable-step inertial-extrapolation rule:

```
θ_k = min(θ, 1 / (k² · ‖x_k - x_{k-1}‖))    if x_k ≠ x_{k-1}
      θ                                       otherwise
```

`θ ∈ (0, 1)`; `0.25` is a common default value in the literature.

The `1/k²` cap ensures ``\\sum_k θ_k \\|x_k - x_{k-1}\\| \\le \\sum_k 1/k^2 < ∞``,
which is the summability condition that drives the convergence proof
for inertial schemes of this class. Lineage: Alvarez & Attouch (2001)
introduced the heavy-ball / inertial idea for monotone operators;
Maingé (2008) gave the modern form for inertial KM-type algorithms;
the specific `1/k²` cap used here is the form adopted by recent
derivative-free projection methods (see References).

```jldoctest
julia> Inertial().θ
0.25

julia> Inertial(0.4).θ
0.4

julia> inertial_coef(Inertial(0.25), 0, [1.0], [0.0])    # k=0: no inertia
0.0
```
"""
struct Inertial <: AbstractInertialRule
    θ::Float64
    function Inertial(θ::Real=0.25)
        (0 < θ < 1) || throw(ArgumentError("Inertial: θ must be in (0, 1)"))
        return new(Float64(θ))
    end
end

function inertial_coef(rule::Inertial, k::Int,
                       xk::AbstractVector, xkm1::AbstractVector)
    k == 0 && return 0.0   # no inertia at k=0 (no x_{-1})
    diff_norm_sq = 0.0
    @inbounds for i in eachindex(xk)
        diff_norm_sq += abs2(xk[i] - xkm1[i])
    end
    diff_norm = sqrt(diff_norm_sq)
    if diff_norm > 0
        return min(rule.θ, 1.0 / (k^2 * diff_norm))
    else
        return rule.θ
    end
end

# ============================================================================
# Apply inertia in place
# ============================================================================

"""
    apply_inertial!(w, rule, k, xk, xkm1) -> (w, θ_k)

Compute ``w = x_k + θ_k (x_k - x_{k-1})`` in place, returning ``θ_k`` too.
"""
function apply_inertial!(w::AbstractVector, rule::AbstractInertialRule,
                         k::Int, xk::AbstractVector, xkm1::AbstractVector)
    θ = inertial_coef(rule, k, xk, xkm1)
    @inbounds @simd for i in eachindex(xk)
        w[i] = xk[i] + θ * (xk[i] - xkm1[i])
    end
    return (w, θ)
end
