# search_directions.jl — Search-direction rules.
#
# A search-direction rule computes d_k from the current and previous
# function values and iterates. Rules are stateless; iteration state
# lives in the Phase 2 cache and is exposed via a per-step context
# NamedTuple to the `direction!` method.

"""
    AbstractSearchDirection

Supertype for search-direction rules. Concrete subtypes implement
the single 3-argument method

```julia
direction!(d, rule, ctx)
```

where the output direction `d` is written in place and `ctx` is a
NamedTuple carrying the per-iteration cache state. The vector fields
alias into the [`DFProjectionCache`](@ref) buffers, parametric on
element type `T`, where `T = eltype(x0) <: AbstractFloat` (with
`Float64` fallback if `eltype(x0)` is not floating). The fields are:

| Field | Type | Description |
|---|---|---|
| `ctx.Fw` | `Vector{T}` | ``F`` at the inertial point ``w_k`` |
| `ctx.Fw_prev` | `Vector{T}` | ``F`` at the previous inertial point ``w_{k-1}`` |
| `ctx.w` | `Vector{T}` | inertial point ``w_k`` |
| `ctx.w_prev` | `Vector{T}` | previous inertial point ``w_{k-1}`` |
| `ctx.d_prev` | `Vector{T}` | previous direction ``d_{k-1}`` |
| `ctx.k` | `Int` | iteration index (0-based) |
| `ctx.α_prev` | `T` | previous line-search step ``\\alpha_{k-1}``; `one(T)` at `k = 0` |

A rule that doesn't need a given field simply ignores it. New cache
fields can be added in future versions without breaking existing
direction methods — they will just be unused by older code.

For `ctx.k == 0`, only `ctx.Fw` is meaningful; the other fields may be
uninitialized. Rules typically set `d = -Fw` at `k = 0`.

The convergence theory for this class of algorithms requires sufficient
descent (`-Fw' d ≥ c‖Fw‖²`) and boundedness (`‖d‖ ≤ c̄‖Fw‖`). It is
the rule author's responsibility to ensure these hold.
"""
abstract type AbstractSearchDirection end

# Default: direction rules are stateless (see init_state contract in types.jl).
init_state(::AbstractSearchDirection, prob, x, alg) = nothing

# ============================================================================
# SpectralThreeTerm: Spectral Three-Term Derivative-Free Projection Method
# (Ibrahim, Alshahrani, Al-Homidan 2023, Numerical Algorithms,
#  https://doi.org/10.1007/s11075-023-01679-7)
# ============================================================================

"""
    SpectralThreeTerm(; r=0.1, alpha_bar=1.0, alpha_min=1e-10, alpha_max=1e30)

Spectral three-term derivative-free direction. The formula is:

```
d_0 = -F(w_0)
d_k = -ϑ_k^I · F(w_k) + β_k · d_{k-1} - ϑ_k^II · y_{k-1}        (k ≥ 1)
```

with

```
y_{k-1} = F(w_k) - F(w_{k-1})
s_{k-1} = (w_k - w_{k-1}) + r · y_{k-1}
ϑ_k^I   = clamp((s_{k-1}' y_{k-1}) / (y_{k-1}' y_{k-1}), alpha_min, alpha_max)
v_k     = max(alpha_bar · ‖d_{k-1}‖ · ‖y_{k-1}‖, ‖F(w_{k-1})‖²)
β_k     = (F(w_k)' y_{k-1}) / v_k
ϑ_k^II  = (F(w_k)' d_{k-1}) / v_k
```

In the degenerate case `y_{k-1} = 0`, ``ϑ_k^I`` falls back to
`alpha_min`, preserving the strict-descent property ``F(w_k)' d_k ≤
-\\alpha_{\\min} \\, \\|F(w_k)\\|^2``.

The uniform bound `alpha_min ≤ ϑ_k^I ≤ alpha_max` is what
underwrites the sufficient-descent and trust-region properties of
this class of algorithms.

# Parameters
- `r`: spectral parameter in the definition of `s_{k-1}` (default 0.1).
- `alpha_bar`: the parameter ``\\bar{α}_1`` in the definition of `v_k`
  (default 1.0).
- `alpha_min`: lower clamp on the spectral coefficient ``ϑ_k^I``
  (default `1e-10`); also the degenerate-case fallback.
- `alpha_max`: upper clamp on the spectral coefficient ``ϑ_k^I``
  (default `1e30`).
"""
Base.@kwdef struct SpectralThreeTerm <: AbstractSearchDirection
    r::Float64         = 0.1
    alpha_bar::Float64 = 1.0
    alpha_min::Float64 = 1e-10
    alpha_max::Float64 = 1e30
end

"""
    direction!(d, rule, ctx)

In-place compute ``d = d_k`` per `rule`. See [`AbstractSearchDirection`](@ref)
for the `ctx` NamedTuple shape.
"""
function direction!(d::AbstractVector, rule::SpectralThreeTerm, ctx)
    Fw      = ctx.Fw
    Fw_prev = ctx.Fw_prev
    w       = ctx.w
    w_prev  = ctx.w_prev
    d_prev  = ctx.d_prev
    k       = ctx.k
    T       = eltype(Fw)

    if k == 0
        @inbounds @simd for i in eachindex(Fw)
            d[i] = -Fw[i]
        end
        return d
    end

    # ── single pass: gather dot products ────────────────────────────────
    #   y_i  = Fw[i] - Fw_prev[i]
    #   s_i  = (w[i] - w_prev[i]) + r · y_i
    yy_sq  = zero(T)   # ‖y‖²
    sy     = zero(T)   # s' y
    Fw_y   = zero(T)   # Fw' y
    Fw_d   = zero(T)   # Fw' d_prev
    dd_sq  = zero(T)   # ‖d_prev‖²
    Fwm_sq = zero(T)   # ‖Fw_prev‖²
    # rule.r and rule.alpha_bar are Float64 hyperparameters (Approach A);
    # coerce to T here for type-stable inner loops.
    r = T(rule.r)
    @inbounds for i in eachindex(Fw)
        yi = Fw[i] - Fw_prev[i]
        si = (w[i] - w_prev[i]) + r * yi
        yy_sq  += yi * yi
        sy     += si * yi
        Fw_y   += Fw[i] * yi
        Fw_d   += Fw[i] * d_prev[i]
        dd_sq  += d_prev[i] * d_prev[i]
        Fwm_sq += Fw_prev[i] * Fw_prev[i]
    end

    yy_norm = sqrt(yy_sq)
    dd_norm = sqrt(dd_sq)

    v_k = max(T(rule.alpha_bar) * dd_norm * yy_norm, Fwm_sq)
    v_k = max(v_k, eps(typeof(v_k)))   # guard against v_k = 0

    α_min_T = T(rule.alpha_min)
    α_max_T = T(rule.alpha_max)
    ϑ_I_raw = yy_sq > 0 ? sy / yy_sq : α_min_T
    ϑ_I     = clamp(ϑ_I_raw, α_min_T, α_max_T)
    β_k  = Fw_y / v_k
    ϑ_II = Fw_d / v_k

    # ── assemble d ──────────────────────────────────────────────────────
    @inbounds @simd for i in eachindex(Fw)
        yi = Fw[i] - Fw_prev[i]
        d[i] = -ϑ_I * Fw[i] + β_k * d_prev[i] - ϑ_II * yi
    end
    return d
end
