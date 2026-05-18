# algorithm.jl — DFProjection algorithm, cache, and step!.
#
# Implements Algorithm 1 (UIDFPAF) of Ibrahim, Alshahrani, Al-Homidan (JOTA 2026).
# One outer iteration:
#   1. Inertial:        w_k = x_k + θ_k (x_k - x_{k-1})
#   2. Eval:            F_w = F(w_k); early-stop if ‖F_w‖ ≤ ε
#   3. Direction:       d_k = direction!(...)
#   4. Backtracking LS: α_k via the seven LS variants
#   5. Trial:           z_k = w_k + α_k d_k;  F_z = F(z_k)
#                       early-stop if ‖F_z‖ ≤ ε AND z_k ∈ X
#   6. Hyperplane:      H_k = {x : F_z' (x - z_k) ≤ 0}
#                       λ_k = F_z' (w_k - z_k) / ‖F_z‖²
#                       ε_k = (ζ²/2) ‖λ_k F_z‖²
#   7. Projection:      x_{k+1} = approx_project(X ∩ H_k, w_k - λ_k F_z, ε_k)

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
                    LS<:LineSearch.AbstractLineSearchAlgorithm,
                    In<:AbstractInertialRule,
                    Stop<:AbstractStoppingCriterion,
                    IU<:AbstractIterateUpdate,
                    CB<:Vector{<:AbstractCallback}} <: AbstractDFProjection
    direction::Dir
    linesearch::LS
    inertial::In
    abstol::Float64
    maxiters::Int
    stopping::Stop
    ζ::Float64
    inner_maxiter::Int
    maxbt::Int
    iterate_update::IU
    callbacks::CB
end

function DFProjection(;
        direction  = SpectralThreeTerm(),
        linesearch = ResidualNormBacktrack(),
        inertial   = Inertial(0.25),
        abstol::Real     = 1e-6,
        maxiters::Int    = 2000,
        stopping::Union{Nothing, AbstractStoppingCriterion} = nothing,
        ζ::Real          = 0.5,
        inner_maxiter::Int = 500,
        maxbt::Int       = 50,
        iterate_update = SolodovSvaiterProjection(),
        callbacks::Vector{<:AbstractCallback} = AbstractCallback[],
    )
    stop = stopping === nothing ?
        AnyOf(AbsResidualTol(Float64(abstol)), MaxIters(maxiters)) :
        stopping
    return DFProjection(direction, linesearch, inertial,
                        Float64(abstol), maxiters, stop,
                        Float64(ζ), inner_maxiter, maxbt,
                        iterate_update, callbacks)
end

# ============================================================================
# DFProjectionCache: mutable state for one solve
# ============================================================================

"""
    DFProjectionCache{Alg, F, DirState, LSCache, IUpState, StopState}

Mutable per-solve state. Created via `init_cache(F, x0, alg)`. All
per-iteration buffers are pre-allocated; `step!` does not allocate.

# Common state (vectors)
- `x`, `x_prev`, `w`, `w_prev`, `z`, `d`, `d_prev` — iteration vectors.
- `Fw`, `Fw_prev`, `Fz` — function values.
- `x_new`, `scratch1` — generic scratch.

# Pluggable component state (typed via parameters)
- `direction_state::DirState` — `init_state(alg.direction, ...)`. Default `nothing`.
- `line_search_cache::LSCache` — populated by the line-search component. Default `nothing` (Stage 2 transition state; filled by Stage 3 once LineSearch.jl is adopted).
- `iterate_update_state::IUpState` — `init_state(...)` for the iterate-update strategy. In Stage 2 this is always a `SolodovSvaiterState` (holding the projection scratch and Dykstra buffers).
- `stopping_state::StopState` — `init_state(alg.stopping, ...)`. Default `nothing`.

# Common scalars
- `k::Int` — iteration counter (0-based).
- `n_evals::Int` — total F evaluations.
- `converged::Bool`, `done::Bool`, `retcode::Symbol`, `resid::Float64`.
- `F0_norm::Float64` — `‖F(x_0)‖`, used by `RelResidualTol`.
- `t_start::Float64` — wall-clock seconds at solve start (for `MaxTime`).
- `α_prev::Float64` — previous line-search step size, surfaced to `direction!` via `ctx.α_prev`.
"""
mutable struct DFProjectionCache{Alg<:DFProjection, F,
                                  S<:AbstractConstraintSet,
                                  DirState, LSCache, IUpState, StopState}
    alg::Alg
    F::F
    set::S    # resolved at init_cache time from the problem (Stage 6)

    # Common per-iteration state
    x::Vector{Float64}
    x_prev::Vector{Float64}
    w::Vector{Float64}
    w_prev::Vector{Float64}
    z::Vector{Float64}
    d::Vector{Float64}
    d_prev::Vector{Float64}
    Fw::Vector{Float64}
    Fw_prev::Vector{Float64}
    Fz::Vector{Float64}
    x_new::Vector{Float64}
    scratch1::Vector{Float64}

    # Pluggable component state — typed via parameters
    direction_state::DirState
    line_search_cache::LSCache
    iterate_update_state::IUpState
    stopping_state::StopState

    # State scalars
    k::Int
    n_evals::Int
    converged::Bool
    done::Bool
    retcode::Symbol
    resid::Float64

    F0_norm::Float64
    t_start::Float64
    α_prev::Float64
end

# Custom show — avoids dumping the long parametric type signature.
function Base.show(io::IO, cache::DFProjectionCache)
    n = length(cache.x)
    print(io, "DFProjectionCache(n=", n, ", k=", cache.k,
              ", retcode=:", cache.retcode, ")")
end

# ============================================================================
# Cache initialization
# ============================================================================

"""
    init_cache(F, x0, alg) -> DFProjectionCache

Allocate per-solve buffers and prepare the initial state. Projects
`x0` onto `set` to ensure feasibility (paper assumes `x_0 ∈ X`).
Evaluates `F` once at the projected `x_0` to populate `F0_norm`
(used by `RelResidualTol`) and starts the wall-clock for `MaxTime`.
"""
function init_cache(F, x0::AbstractVector, alg::DFProjection;
                     set::AbstractConstraintSet = RealSpace())
    n = length(x0)

    x       = Vector{Float64}(undef, n)
    project!(x, x0, set)                    # ensure feasibility (Stage 6: set from problem)

    x_prev  = copy(x)                       # k=0 doesn't use x_prev
    w       = similar(x)
    w_prev  = copy(x)                       # for k=0 SpectralThreeTerm doesn't use this
    z       = similar(x)
    d       = zeros(n)
    d_prev  = zeros(n)
    Fw      = zeros(n)
    Fw_prev = zeros(n)
    Fz      = zeros(n)
    x_new   = similar(x)
    scratch1 = similar(x)

    # One initial F-eval (needed for F0_norm and as the `fu` argument to
    # CommonSolve.init for the line search).
    Fx0 = F(x)
    s = 0.0
    @inbounds for i in eachindex(Fx0)
        s += Fx0[i] * Fx0[i]
    end
    F0_norm = sqrt(s)

    # Pluggable component state. All four go through init_state — defaults
    # are `nothing` for components that don't need scratch
    # (SpectralThreeTerm, DirectUpdate, most stopping criteria);
    # SolodovSvaiterProjection allocates its Dykstra buffers; HalpernUpdate
    # stores x_0.
    dir_state            = init_state(alg.direction,      nothing, x0, alg)
    iterate_update_state = init_state(alg.iterate_update, nothing, x0, alg)
    stopping_state       = init_state(alg.stopping,       nothing, x0, alg)

    # Line-search cache via LineSearch.jl's CommonSolve.init contract.
    # Synthesize a NonlinearProblem from the F closure (test-friendly path).
    # The standard SciML route goes through DFSciMLCache which already has
    # the real prob; that path bypasses this synthesizing.
    f_2arg(u, p) = F(u)
    prob_synth = SciMLBase.NonlinearProblem(f_2arg, x)
    line_search_cache = CommonSolve.init(prob_synth, alg.linesearch, Fx0, x)

    cache = DFProjectionCache(
        alg, F, set,
        x, x_prev, w, w_prev, z, d, d_prev,
        Fw, Fw_prev, Fz, x_new, scratch1,
        dir_state, line_search_cache, iterate_update_state, stopping_state,
        0,         # k
        1,         # n_evals (the F(x_0) eval just done)
        false,     # converged
        false,     # done
        :Default,  # retcode
        NaN,       # resid
        F0_norm,
        time(),    # t_start
        1.0,       # α_prev (ignored at k=0; first line search overwrites)
    )

    # Fire :initialize event before returning the cache.
    _fire!(cache, alg, :initialize)

    return cache
end

# ============================================================================
# Helpers
# ============================================================================

# Copy z and F(z) from the line-search cache into `cache.z` / `cache.Fz`,
# and credit any F evaluations the line search did. Fast path when the
# inner cache exposes our DFMethods fields (z_cache, fu_cache, n_evals);
# fallback recomputes F(z) for ecosystem line-search caches that don't.
@inline function _harvest_linesearch_state!(cache, α)
    ls = cache.line_search_cache
    if hasfield(typeof(ls), :z_cache) && hasfield(typeof(ls), :fu_cache) &&
       hasfield(typeof(ls), :n_evals)
        copyto!(cache.z,  ls.z_cache)
        copyto!(cache.Fz, ls.fu_cache)
        cache.n_evals += ls.n_evals
    else
        # Ecosystem line search — recompute z, F(z), eat one extra F-eval.
        @inbounds @simd for j in eachindex(cache.w)
            cache.z[j] = cache.w[j] + α * cache.d[j]
        end
        Fz_value = cache.F(cache.z)
        @inbounds @simd for j in eachindex(cache.Fz)
            cache.Fz[j] = Fz_value[j]
        end
        cache.n_evals += 1
    end
    return nothing
end

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

# Fire `event` to all observers and the stopping criterion. Sets cache.done
# if any stopping criterion fires.
function _fire!(cache::DFProjectionCache, alg::DFProjection, event::Symbol)
    @inbounds for cb in alg.callbacks
        on_event!(cb, cache, event)
    end
    stopped, code = on_event!(alg.stopping, cache, event)
    if stopped && !cache.done
        cache.done      = true
        cache.retcode   = code
        cache.converged = (code === :Success)
    end
    return nothing
end

"""
    step!(cache::DFProjectionCache) -> cache

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

    # ── Step 2: F(w_k) ────────────────────────────────────────────────────────
    Fw_value = F(cache.w)
    @inbounds @simd for i in eachindex(cache.Fw)
        cache.Fw[i] = Fw_value[i]
    end
    cache.n_evals += 1

    # ── Step 3: search direction d_k ──────────────────────────────────────────
    ctx = (;
        Fw      = cache.Fw,
        Fw_prev = cache.Fw_prev,
        w       = cache.w,
        w_prev  = cache.w_prev,
        d_prev  = cache.d_prev,
        k       = k,
        α_prev  = cache.α_prev,
    )
    direction!(cache.d, alg.direction, ctx)

    # ── Step 4: line search → α_k, z_k, F(z_k) ───────────────────────────────
    ls_sol = CommonSolve.solve!(cache.line_search_cache, cache.w, cache.d)
    α  = ls_sol.step_size
    ok = ls_sol.retcode === SciMLBase.ReturnCode.Success
    _harvest_linesearch_state!(cache, α)

    if !ok
        cache.resid   = _norm2(cache.Fw)
        cache.retcode = :LineSearchFailed
        cache.done    = true
        return cache
    end

    # ── Fire :post_linesearch (residual-based stopping checks fire here) ─────
    _fire!(cache, alg, :post_linesearch)
    if cache.done
        # Stopping fired on cache.Fz — z is the converged point (if feasible)
        if cache.retcode === :Success
            if _is_feasible(cache.z, cache.set, cache.scratch1)
                copyto!(cache.x, cache.z)
                cache.resid = _norm2(cache.Fz)
            else
                # Infeasible z — back out the stop; let the iter proceed
                cache.done      = false
                cache.retcode   = :Default
                cache.converged = false
            end
        else
            # Non-Success stop (e.g., user-defined) — accept z if feasible,
            # else accept current x.
            if _is_feasible(cache.z, cache.set, cache.scratch1)
                copyto!(cache.x, cache.z)
                cache.resid = _norm2(cache.Fz)
            else
                cache.resid = _norm2(cache.Fw)
            end
        end
        cache.done && return cache
    end

    # Degeneracy guard: F(z) ≈ 0 but no stopping criterion claimed :Success.
    # Without an explicit convergence tolerance pinned to this case, we have
    # no λ_k to project — declare a degenerate residual.
    Fz_norm = _norm2(cache.Fz)
    if Fz_norm < eps()
        cache.resid   = Fz_norm
        cache.retcode = :DegenerateResidual
        cache.done    = true
        return cache
    end

    # ── Steps 5–6: delegate to the iterate-update strategy ───────────────────
    update_ctx = (; w     = cache.w,
                    d     = cache.d,
                    α     = α,
                    z     = cache.z,
                    Fw    = cache.Fw,
                    Fz    = cache.Fz,
                    set   = cache.set,
                    k     = k,
                    ζ     = alg.ζ,
                    inner_maxiter = alg.inner_maxiter,
                    state = cache.iterate_update_state)
    update_iterate!(cache.x_new, alg.iterate_update, update_ctx)

    # ── Step 7: shift state for next iteration ────────────────────────────────
    copyto!(cache.x_prev,    cache.x)
    copyto!(cache.x,         cache.x_new)
    copyto!(cache.w_prev,    cache.w)
    copyto!(cache.Fw_prev,   cache.Fw)
    copyto!(cache.d_prev,    cache.d)
    cache.α_prev = α

    cache.k += 1

    # ── Fire :post_iter (iteration-budget + stall criteria fire here) ────────
    _fire!(cache, alg, :post_iter)
    # resid is left as NaN for end-of-iter stops; the driver finalizes it
    # with a single F(x_final) call after the loop.

    return cache
end

# Note: the inner-cache run-loop has moved into
# `CommonSolve.solve!(::DFSciMLCache)` in `nonlinearsolve_integration.jl`.
# `DFProjectionCache` is driven externally via repeated `step!` calls;
# there is no longer a standalone `solve_df` / `solve_df!` entry point.
