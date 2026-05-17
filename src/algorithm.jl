# algorithm.jl — DFProjection algorithm, cache, step!, and solve_df driver.
#
# Implements Algorithm 1 (UIDFPAF) of Ibrahim, Alshahrani, Al-Homidan (JOTA 2026).
# One outer iteration:
#   1. Inertial:        w_k = x_k + θ_k (x_k - x_{k-1})
#   2. Eval:            ψ_w = ψ(w_k); early-stop if ‖ψ_w‖ ≤ ε
#   3. Direction:       d_k = direction!(...)
#   4. Backtracking LS: α_k via the seven LS variants
#   5. Trial:           z_k = w_k + α_k d_k;  ψ_z = ψ(z_k)
#                       early-stop if ‖ψ_z‖ ≤ ε AND z_k ∈ X
#   6. Hyperplane:      H_k = {x : ψ_z' (x - z_k) ≤ 0}
#                       λ_k = ψ_z' (w_k - z_k) / ‖ψ_z‖²
#                       ε_k = (ζ²/2) ‖λ_k ψ_z‖²
#   7. Projection:      x_{k+1} = approx_project(X ∩ H_k, w_k - λ_k ψ_z, ε_k)

# ============================================================================
# DFProjection: concrete algorithm
# ============================================================================

"""
    DFProjection(; direction, linesearch, inertial, set,
                   abstol, maxiters, stopping, ζ, inner_maxiter, maxbt)

Concrete derivative-free projection algorithm. Each component is
swappable; defaults reproduce the Ibrahim 2026 paper setup (SpectralThreeTerm
direction, LSII line search, `Inertial(0.25)`, paper `ζ = 0.5`).

# Fields
- `direction::AbstractSearchDirection` — search-direction rule. Default: `SpectralThreeTerm()`.
- `linesearch::AbstractDFLineSearch`   — line search. Default: `LSII()`.
- `inertial::AbstractInertialRule`     — inertial rule. Default: `Inertial(0.25)`.
- `set::AbstractConstraintSet`         — feasible set ``X``. Default: `RealSpace()`.
- `abstol::Float64`                    — residual tolerance used to build the default stopping. Default: `1e-6`.
- `maxiters::Int`                      — outer-iteration cap used to build the default stopping. Default: `2000`.
- `stopping::AbstractStoppingCriterion` — full stopping rule. If not supplied, built as `AnyOf(AbsResidualTol(abstol), MaxIters(maxiters))`.
- `ζ::Float64`                         — approximate-projection tolerance factor (paper ζ_k ≡ ζ). Default: `0.5`.
- `inner_maxiter::Int`                 — max Dykstra iterations inside `approx_project_X_halfspace!`. Default: `500`.
- `maxbt::Int`                         — max line-search backtracks per iteration. Default: `50`.

# Stopping criteria
`abstol` and `maxiters` are convenience knobs that build the default
stopping rule. For composite or domain-specific criteria, pass
`stopping = AnyOf(RelResidualTol(...), StepNormTol(...), MaxTime(...), …)`;
the `abstol`/`maxiters` fields are still stored (for introspection and
SciMLBase kwarg overrides) but `step!` ignores them in favour of the
supplied `stopping`.
"""
struct DFProjection{Dir<:AbstractSearchDirection,
                    LS<:AbstractDFLineSearch,
                    In<:AbstractInertialRule,
                    Set<:AbstractConstraintSet,
                    Stop<:AbstractStoppingCriterion} <: AbstractDFProjectionAlgorithm
    direction::Dir
    linesearch::LS
    inertial::In
    set::Set
    abstol::Float64
    maxiters::Int
    stopping::Stop
    ζ::Float64
    inner_maxiter::Int
    maxbt::Int
end

function DFProjection(;
        direction  = SpectralThreeTerm(),
        linesearch = LSII(),
        inertial   = Inertial(0.25),
        set        = RealSpace(),
        abstol::Real     = 1e-6,
        maxiters::Int    = 2000,
        stopping::Union{Nothing, AbstractStoppingCriterion} = nothing,
        ζ::Real          = 0.5,
        inner_maxiter::Int = 500,
        maxbt::Int       = 50,
    )
    stop = stopping === nothing ?
        AnyOf(AbsResidualTol(Float64(abstol)), MaxIters(maxiters)) :
        stopping
    return DFProjection(direction, linesearch, inertial, set,
                        Float64(abstol), maxiters, stop,
                        Float64(ζ), inner_maxiter, maxbt)
end

# ============================================================================
# DFProjectionCache: mutable state for one solve
# ============================================================================

"""
    DFProjectionCache

Mutable per-solve state. Created via `init_cache(F, x0, alg)`. All
per-iteration buffers are pre-allocated; `step!` does not allocate.

State fields:
- `x`, `x_prev`, `w`, `w_prev`, `z`, `d`, `d_prev` — iteration vectors.
- `ψw`, `ψw_prev`, `ψz` — function values.
- `x_new`, `proj_target` — buffers for the projection step.
- `proj_p`, `proj_q`, `proj_scratch`, `proj_out_prev` — Dykstra inner buffers.
- `k::Int` — iteration counter (0-based).
- `n_evals::Int` — total ψ evaluations.
- `converged::Bool` — true on successful early-stop.
- `done::Bool` — true once `solve_df!` should exit the loop.
- `retcode::Symbol` — `:Default`, `:Success`, `:MaxIters`, `:LineSearchFailed`,
  `:DegenerateResidual`.
- `resid::Float64` — final ‖ψ‖ at the returned iterate (`NaN` until set).
"""
mutable struct DFProjectionCache{Alg<:DFProjection, F}
    alg::Alg
    F::F

    # State vectors
    x::Vector{Float64}
    x_prev::Vector{Float64}
    w::Vector{Float64}
    w_prev::Vector{Float64}
    z::Vector{Float64}
    d::Vector{Float64}
    d_prev::Vector{Float64}
    ψw::Vector{Float64}
    ψw_prev::Vector{Float64}
    ψz::Vector{Float64}
    x_new::Vector{Float64}
    proj_target::Vector{Float64}

    # Projection inner buffers
    proj_p::Vector{Float64}
    proj_q::Vector{Float64}
    proj_scratch::Vector{Float64}
    proj_out_prev::Vector{Float64}

    # Generic scratch
    scratch1::Vector{Float64}

    # State scalars
    k::Int
    n_evals::Int
    converged::Bool
    done::Bool
    retcode::Symbol
    resid::Float64

    # Stopping-criterion state
    ψ0_norm::Float64        # ‖ψ(x_0)‖, set by `init_cache` (for RelResidualTol)
    t_start::Float64        # wall-clock seconds at solve start (for MaxTime)

    # Previous line-search step size α_{k-1}. Surfaced to `direction!` via
    # the `ctx.α_prev` field (see `search_directions.jl`). Initialized to
    # 1.0 by `init_cache`; updated by `step!` after each successful line
    # search. Used by directions whose update rule depends on the previous
    # step displacement s_{k-1} = α_{k-1} · d_{k-1} (e.g. GMOPCGM).
    α_prev::Float64
end

# ============================================================================
# DFSolution: returned from solve_df
# ============================================================================

"""
    DFSolution

Immutable result returned from `solve_df`. Phase 3 will replace this
with a `SciMLBase.NonlinearSolution` once `NonlinearSolveBase` is wired
in; the field set will be preserved.
"""
struct DFSolution
    x::Vector{Float64}
    resid::Float64
    converged::Bool
    iterations::Int
    n_evals::Int
    retcode::Symbol
end

# ============================================================================
# Cache initialization
# ============================================================================

"""
    init_cache(F, x0, alg) -> DFProjectionCache

Allocate per-solve buffers and prepare the initial state. Projects
`x0` onto `alg.set` to ensure feasibility (paper assumes `x_0 ∈ X`).
Evaluates `ψ` once at the projected `x_0` to populate `ψ0_norm`
(used by `RelResidualTol`) and starts the wall-clock for `MaxTime`.
"""
function init_cache(F, x0::AbstractVector, alg::DFProjection)
    n = length(x0)

    x       = Vector{Float64}(undef, n)
    project!(x, x0, alg.set)                # ensure feasibility

    x_prev  = copy(x)                       # k=0 doesn't use x_prev
    w       = similar(x)
    w_prev  = copy(x)                       # for k=0 SpectralThreeTerm doesn't use this
    z       = similar(x)
    d       = zeros(n)
    d_prev  = zeros(n)
    ψw      = zeros(n)
    ψw_prev = zeros(n)
    ψz      = zeros(n)
    x_new   = similar(x)
    proj_target   = similar(x)
    proj_p        = zeros(n)
    proj_q        = zeros(n)
    proj_scratch  = similar(x)
    proj_out_prev = similar(x)
    scratch1      = similar(x)

    # One initial ψ-eval for ψ0_norm (used by `RelResidualTol`).
    ψx0 = F(x)
    s = 0.0
    @inbounds for i in eachindex(ψx0)
        s += ψx0[i] * ψx0[i]
    end
    ψ0_norm = sqrt(s)

    return DFProjectionCache(
        alg, F,
        x, x_prev, w, w_prev, z, d, d_prev,
        ψw, ψw_prev, ψz,
        x_new, proj_target,
        proj_p, proj_q, proj_scratch, proj_out_prev,
        scratch1,
        0,         # k
        1,         # n_evals (the ψ(x_0) eval just done)
        false,     # converged
        false,     # done
        :Default,  # retcode
        NaN,       # resid
        ψ0_norm,   # ψ0_norm
        time(),    # t_start
        1.0,       # α_prev (ignored at k=0; first line search overwrites)
    )
end

# ============================================================================
# Helpers
# ============================================================================

@inline function _norm2(v::AbstractVector)
    s = 0.0
    @inbounds for i in eachindex(v)
        s += v[i] * v[i]
    end
    return sqrt(s)
end

# Check whether x is feasible for set (within tol) using a projection probe.
function _is_feasible(x::AbstractVector, set::AbstractConstraintSet,
                      scratch::AbstractVector; tol::Float64 = 1e-10)
    project!(scratch, x, set)
    diff_sq = 0.0
    @inbounds for i in eachindex(x)
        diff_sq += abs2(scratch[i] - x[i])
    end
    return sqrt(diff_sq) <= tol
end

# ============================================================================
# step!: one outer iteration of Algorithm 1
# ============================================================================

"""
    step!(cache) -> cache

Perform one outer iteration of UIDFPAF. Updates `cache.x`, `cache.k`,
function-value buffers, and sets `cache.done = true` on termination.
Returns the cache (mutated).
"""
function step!(cache::DFProjectionCache)
    cache.done && return cache

    alg = cache.alg
    F   = cache.F
    k   = cache.k

    # ── Step 1: inertial point w_k = x_k + θ_k (x_k - x_{k-1}) ───────────────
    apply_inertial!(cache.w, alg.inertial, k, cache.x, cache.x_prev)

    # ── Step 2: ψ(w_k) ────────────────────────────────────────────────────────
    ψw_value = F(cache.w)
    @inbounds @simd for i in eachindex(cache.ψw)
        cache.ψw[i] = ψw_value[i]
    end
    cache.n_evals += 1

    # ── Step 3: stopping criteria after ψ(w_k) ───────────────────────────────
    stopped, code = should_stop_at_w(alg.stopping, cache)
    if stopped
        copyto!(cache.x, cache.w)
        cache.resid     = _norm2(cache.ψw)
        cache.converged = (code === :Success)
        cache.retcode   = code
        cache.done      = true
        return cache
    end

    # ── Step 4: search direction d_k ──────────────────────────────────────────
    # Build a per-iteration context NamedTuple. Zero-allocation in practice
    # (stack-allocated). Direction rules pull whichever fields they need.
    ctx = (;
        ψw      = cache.ψw,
        ψw_prev = cache.ψw_prev,
        w       = cache.w,
        w_prev  = cache.w_prev,
        d_prev  = cache.d_prev,
        k       = k,
        α_prev  = cache.α_prev,
    )
    direction!(cache.d, alg.direction, ctx)

    # ── Step 5: backtracking line search → α_k, z_k, ψ(z_k) ──────────────────
    α, _, n_evals_ls, ok = linesearch!(alg.linesearch, F,
                                       cache.w, cache.d,
                                       cache.z, cache.ψz;
                                       maxbt = alg.maxbt)
    cache.n_evals += n_evals_ls

    if !ok
        # Line search failed — report best-effort residual at w_k
        cache.resid   = _norm2(cache.ψw)
        cache.retcode = :LineSearchFailed
        cache.done    = true
        return cache
    end

    # ── Step 6: stopping criteria after ψ(z_k) (z_k ∈ X required for Success) ──
    stopped, code = should_stop_at_z(alg.stopping, cache)
    if stopped
        if _is_feasible(cache.z, alg.set, cache.scratch1)
            copyto!(cache.x, cache.z)
            cache.resid     = _norm2(cache.ψz)
            cache.converged = (code === :Success)
            cache.retcode   = code
            cache.done      = true
            return cache
        end
    end

    # Degeneracy guard: λ_k = ψz'(w-z) / ‖ψz‖² needs ‖ψz‖ > 0.
    ψz_norm = _norm2(cache.ψz)
    if ψz_norm < eps()
        cache.resid   = ψz_norm
        cache.retcode = :DegenerateResidual
        cache.done    = true
        return cache
    end

    # ── Step 7: hyperplane H_k and approximate projection ─────────────────────
    # H_k = {x : ψz' (x - z_k) ≤ 0}  →  a = ψz, c = ψz' z_k
    # λ_k = ψz'(w_k - z_k) / ‖ψz‖²
    inner_wz   = 0.0
    inner_ψz_z = 0.0
    ψz_norm_sq = ψz_norm * ψz_norm
    @inbounds for i in eachindex(cache.w)
        inner_wz   += cache.ψz[i] * (cache.w[i] - cache.z[i])
        inner_ψz_z += cache.ψz[i] * cache.z[i]
    end
    λ = inner_wz / ψz_norm_sq

    # target = w_k - λ ψz
    @inbounds @simd for i in eachindex(cache.w)
        cache.proj_target[i] = cache.w[i] - λ * cache.ψz[i]
    end

    # ε_k = (ζ²/2) ‖λ ψz‖² = (ζ²/2) λ² ‖ψz‖²
    ε_k = 0.5 * alg.ζ^2 * λ * λ * ψz_norm_sq

    # Project onto X ∩ H_k
    approx_project_X_halfspace!(cache.x_new, cache.proj_target,
                                alg.set,
                                cache.ψz, inner_ψz_z, ε_k,
                                cache.proj_p, cache.proj_q,
                                cache.proj_scratch, cache.proj_out_prev;
                                maxiter = alg.inner_maxiter)

    # ── Step 8: shift state for next iteration ────────────────────────────────
    copyto!(cache.x_prev,    cache.x)
    copyto!(cache.x,         cache.x_new)
    copyto!(cache.w_prev,    cache.w)
    copyto!(cache.ψw_prev,   cache.ψw)
    copyto!(cache.d_prev,    cache.d)
    cache.α_prev = α

    cache.k += 1

    # ── Step 9: stopping criteria at end of iteration ─────────────────────────
    stopped, code = should_stop_at_end(alg.stopping, cache)
    if stopped
        cache.converged = (code === :Success)
        cache.retcode   = code
        cache.done      = true
        # resid is left as NaN for end-of-iter stops; `solve_df!`
        # finalizes it with a single ψ(x_final) call after the loop.
    end

    return cache
end

# ============================================================================
# Driver: solve_df / solve_df!
# ============================================================================

"""
    solve_df!(cache) -> cache

Drive the algorithm to termination given an initialized `cache`.
"""
function solve_df!(cache::DFProjectionCache)
    while !cache.done
        step!(cache)
    end
    # Finalize residual if the loop hit max-iters or line-search failure
    # without setting it via an early-return branch.
    if isnan(cache.resid)
        ψx_final = cache.F(cache.x)
        s = 0.0
        @inbounds for i in eachindex(ψx_final)
            s += ψx_final[i] * ψx_final[i]
        end
        cache.resid    = sqrt(s)
        cache.n_evals += 1
    end
    return cache
end

"""
    solve_df(F, x0, alg) -> DFSolution

Top-level entry point. Builds a cache, runs the loop, returns a
`DFSolution`. `F` is an out-of-place mapping `F(x) -> Vector`. Phase 3
will replace this with `SciMLBase.solve(prob::NonlinearProblem, alg)`.

# Example
```julia
F = x -> x .- [0.3, -0.2]   # solution at [0.3, -0.2]
alg = DFProjection(; set = BoxSet([-1.0, -1.0], [1.0, 1.0]),
                    abstol = 1e-8)
sol = solve_df(F, [1.0, -1.0], alg)
sol.x         # ≈ [0.3, -0.2]
sol.converged # true
sol.retcode   # :Success
```
"""
function solve_df(F, x0::AbstractVector, alg::DFProjection)
    cache = init_cache(F, x0, alg)
    solve_df!(cache)
    return DFSolution(
        copy(cache.x),
        cache.resid,
        cache.converged,
        cache.k,
        cache.n_evals,
        cache.retcode,
    )
end
