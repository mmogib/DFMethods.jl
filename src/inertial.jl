# inertial.jl — Inertial extrapolation rules.

"""
    AbstractInertialRule

Supertype for inertial-extrapolation rules. A rule produces the inertial
coefficient ``θ_k`` from the iteration index, current iterate, and
previous iterate. The inertial point is then

```
w_k = x_k + θ_k (x_k - x_{k-1})
```

(Ibrahim 2026 eq. (2)).
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
the paper's accelerated variant.
"""
struct NoInertial <: AbstractInertialRule end

inertial_coef(::NoInertial, k, xk, xkm1) = 0.0

# ============================================================================
# Inertial: Ibrahim 2026 eq. (2)
# ============================================================================

"""
    Inertial(θ = 0.25)

Standard summable-step inertial-extrapolation rule:

```
θ_k = min(θ, 1 / (k² · ‖x_k - x_{k-1}‖))    if x_k ≠ x_{k-1}
      θ                                       otherwise
```

`θ ∈ (0, 1)`. Ibrahim 2026 uses `θ = 0.25` in its experiments.

The `1/k²` cap ensures ``\\sum_k θ_k \\|x_k - x_{k-1}\\| \\le \\sum_k 1/k^2 < ∞``,
which is the summability condition that drives the convergence proof
(Ibrahim 2026 Remark 2.2). Lineage: Alvarez–Attouch 2001 introduced
the heavy-ball / inertial idea for monotone operators; Maingé 2008
gave the modern form; Abubakar et al. 2021 (ref [1] in Ibrahim 2026)
introduced this specific `1/k²` cap for DF projection methods;
Ibrahim 2026 carries it into eq. (2) of the unified framework.
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
