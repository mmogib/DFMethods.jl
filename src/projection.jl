# projection.jl — Approximate projection onto X ∩ H_k.
#
# The Solodov–Svaiter projection step projects the target
#     t = w_k - λ_k F(z_k)
# onto X ∩ H_k where
#     H_k = {x : F(z_k)' (x - z_k) ≤ 0}
# to tolerance ε_k = (ζ²/2) ‖λ_k F(z_k)‖².
#
# We use Dykstra's algorithm over the two sets (X, H_k) with an ε-style
# stopping criterion `‖y_new - y_prev‖² ≤ ε`. To avoid allocating a
# HalfSpace struct each outer iteration, we accept the hyperplane
# normal `a` and offset `c` directly and inline the halfspace projection.

# Note: `SolodovSvaiterState` moved to `iterate_updates.jl` in Stage 4 —
# it's the per-solve state for the SolodovSvaiterProjection strategy, not
# a generic projection-step concern. `approx_project_X_halfspace!` below
# remains as a shared helper that the strategy calls.

"""
    approx_project_X_halfspace!(out, target, X, a, c, ε,
                                p_buf, q_buf, scratch, out_prev;
                                maxiter=500) -> out

Approximate projection of `target` onto ``X \\cap H`` with
``H = \\{x : a^\\top x \\le c\\}``, via Dykstra. Inner buffers
(`p_buf`, `q_buf`, `scratch`, `out_prev`) are pre-allocated by the
caller (typically the algorithm cache). Stops when
``\\|y - y_{\\text{prev}}\\|^2 \\le \\epsilon`` or `maxiter` reached.
"""
function approx_project_X_halfspace!(out::AbstractVector,
                                     target::AbstractVector,
                                     X::AbstractConstraintSet,
                                     a::AbstractVector,
                                     c::Real,
                                     ε::Real,
                                     p_buf::AbstractVector,
                                     q_buf::AbstractVector,
                                     scratch::AbstractVector,
                                     out_prev::AbstractVector;
                                     maxiter::Int = 500)
    fill!(p_buf, 0.0)
    fill!(q_buf, 0.0)
    copyto!(out, target)

    # Cache ‖a‖² for the inline halfspace projection
    a_norm_sq = 0.0
    @inbounds for i in eachindex(a)
        a_norm_sq += a[i] * a[i]
    end
    a_norm_sq <= 0 && throw(ArgumentError("approx_project_X_halfspace!: ‖a‖ = 0"))

    @inbounds for _ in 1:maxiter
        copyto!(out_prev, out)

        # ── Project onto X (with Dykstra correction p_buf) ───────────────
        @. scratch = out + p_buf
        project!(out, scratch, X)
        @. p_buf = scratch - out

        # ── Project onto H = {x : a' x ≤ c} (with Dykstra correction q_buf) ─
        @. scratch = out + q_buf
        s = 0.0
        for i in eachindex(scratch)
            s += a[i] * scratch[i]
        end
        if s <= c
            copyto!(out, scratch)
        else
            t = (s - c) / a_norm_sq
            @simd for i in eachindex(scratch)
                out[i] = scratch[i] - t * a[i]
            end
        end
        @. q_buf = scratch - out

        # ── Convergence: ‖out - out_prev‖² ≤ ε ──────────────────────────
        diff_sq = 0.0
        for i in eachindex(out)
            diff_sq += abs2(out[i] - out_prev[i])
        end
        diff_sq <= ε && break
    end
    return out
end

# ============================================================================
# Specialization: X = RealSpace — no Dykstra, just project onto halfspace
# ============================================================================

function approx_project_X_halfspace!(out::AbstractVector,
                                     target::AbstractVector,
                                     ::RealSpace,
                                     a::AbstractVector,
                                     c::Real,
                                     ε::Real,
                                     p_buf::AbstractVector,
                                     q_buf::AbstractVector,
                                     scratch::AbstractVector,
                                     out_prev::AbstractVector;
                                     maxiter::Int = 500)
    # Exact projection onto halfspace only.
    a_norm_sq = 0.0
    s = 0.0
    @inbounds for i in eachindex(a)
        a_norm_sq += a[i] * a[i]
        s += a[i] * target[i]
    end
    a_norm_sq <= 0 && throw(ArgumentError("approx_project_X_halfspace!: ‖a‖ = 0"))
    if s <= c
        copyto!(out, target)
    else
        t = (s - c) / a_norm_sq
        @inbounds @simd for i in eachindex(target)
            out[i] = target[i] - t * a[i]
        end
    end
    return out
end
