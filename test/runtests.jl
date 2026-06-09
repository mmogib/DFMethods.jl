using DFMethods
using Test
using LinearAlgebra
using Random
using SciMLBase
using CommonSolve
using LineSearch

# Internal helpers — not part of the public API, but exercised by tests.
using DFMethods: inertial_coef, apply_inertial!, approx_project_X_halfspace!

# Recorder direction for testing α_prev plumbing in `step!`. Captures the
# α_prev value that `step!` passes through `ctx` to `direction!(d, rule, ctx)`.
struct _RecorderDir <: AbstractSearchDirection end
const _RECORDED_ALPHA_PREV = Ref{Float64}(NaN)
function DFMethods.direction!(d, ::_RecorderDir, ctx)
    _RECORDED_ALPHA_PREV[] = ctx.α_prev
    @. d = -ctx.Fw   # steepest descent — always valid
    return d
end

# Stateful direction for testing init_state → ctx.direction_state plumbing
# (Phase 2). init_state allocates a call counter; direction! increments it
# via ctx.direction_state; the test asserts the count via
# cache.inner.direction_state. Would have failed on v0.2 (ctx had no
# `direction_state` field).
struct _CountingDir <: AbstractSearchDirection end
mutable struct _CountingState
    calls::Int
end
DFMethods.init_state(::_CountingDir, prob, x0, alg) = _CountingState(0)
function DFMethods.direction!(d, ::_CountingDir, ctx)
    ctx.direction_state.calls += 1
    @. d = -ctx.Fw   # steepest descent
    return d
end

# Schema-pin sentinels: capture `keys(ctx)` so the corresponding testsets
# can assert that the live ctx field-sets still match docs/src/extending.md.
struct _DirCtxCapture <: AbstractSearchDirection end
const _CAPTURED_DIR_CTX = Ref{Tuple}(())
function DFMethods.direction!(d, ::_DirCtxCapture, ctx)
    _CAPTURED_DIR_CTX[] = keys(ctx)
    @. d = -ctx.Fw   # steepest descent
    return d
end

struct _IUCtxCapture <: AbstractIterateUpdate end
const _CAPTURED_IU_CTX = Ref{Tuple}(())
function DFMethods.update_iterate!(x_new, ::_IUCtxCapture, ctx)
    _CAPTURED_IU_CTX[] = keys(ctx)
    @. x_new = ctx.z   # accept trial point as next iterate
    return x_new
end

@testset "DFMethods.jl" begin

    # ========================================================================
    # Constraint sets
    # ========================================================================

    @testset "Constraint sets" begin
        @testset "RealSpace" begin
            x = [1.0, -2.0, 3.0]
            y = similar(x)
            project!(y, x, RealSpace())
            @test y == x
            @test project(x, RealSpace()) == x
        end

        @testset "BoxSet" begin
            set = BoxSet(fill(-1.0, 3), fill(1.0, 3))
            @test project([-2.0, 0.5, 5.0], set) == [-1.0, 0.5, 1.0]

            set2 = BoxSet([0.0, -1.0, -2.0], [1.0, 1.0, 5.0])
            @test project([0.5, 2.0, -3.0], set2) == [0.5, 1.0, -2.0]

            @test_throws ArgumentError BoxSet([0.0, 1.0], [0.0])
            @test_throws ArgumentError BoxSet([1.0, 0.0], [0.0, 1.0])
        end

        @testset "HalfSpace" begin
            hs = HalfSpace([1.0, 1.0], 1.0)
            @test project([0.3, 0.3], hs) ≈ [0.3, 0.3]
            y = project([1.0, 1.0], hs)
            @test sum(y) ≈ 1.0 atol=1e-12
            @test y ≈ [0.5, 0.5] atol=1e-12

            @test_throws ArgumentError HalfSpace([0.0, 0.0], 1.0)
        end

        @testset "CappedBox (polyhedral Ω)" begin
            ω = CappedBox(-1.0, 1.0, 1.0)
            # Case 1: x already inside Ω (sum 0.7 ≤ 1)
            @test project([0.3, 0.4], ω) ≈ [0.3, 0.4]
            # Case 2: box-clamp alone suffices (clamped sum 0.5 ≤ 1)
            @test project([2.0, -0.5], ω) ≈ [1.0, -0.5]
            # Case 3: Lagrange correction needed. Symmetric input → symmetric output
            y3 = project([2.0, 2.0], ω)
            @test sum(y3) ≈ 1.0 atol=1e-10
            @test y3 ≈ [0.5, 0.5] atol=1e-10
            # Case 4: nontrivial — one component pinned at b, the other corrected
            y4 = project([10.0, 0.2], ω)
            @test all(-1.0 .<= y4 .<= 1.0)
            @test sum(y4) <= 1.0 + 1e-10
            # Validation
            @test_throws ArgumentError CappedBox(1.0, 0.0, 0.0)
            # Infeasible (n·a = 0 > c = -1)
            ω_infeas = CappedBox(0.0, 1.0, -1.0)
            @test_throws ArgumentError project!(zeros(2), zeros(2), ω_infeas)
        end

        @testset "Intersection (Dykstra)" begin
            box = BoxSet([-1.0, -1.0], [1.0, 1.0])
            hs  = HalfSpace([1.0, 1.0], 1.0)
            inter = Intersection(box, hs; maxiter=500, tol=1e-12)

            y = project([2.0, 2.0], inter)
            @test all(-1.0 .<= y .<= 1.0)
            @test sum(y) <= 1.0 + 1e-8
            @test y ≈ [0.5, 0.5] atol=1e-6
        end

        @testset "UserSet (l2 unit ball)" begin
            proj_ball!(y, x) = (n = norm(x); y .= (n <= 1) ? x : x ./ n; return y)
            us = UserSet(proj_ball!)
            @test project([0.5, 0.5], us) ≈ [0.5, 0.5]
            y = project([3.0, 4.0], us)
            @test norm(y) ≈ 1.0 atol=1e-12
        end
    end

    # ========================================================================
    # Inertial rules
    # ========================================================================

    @testset "Inertial rules" begin
        @testset "NoInertial" begin
            r = NoInertial()
            @test inertial_coef(r, 5, [1.0, 1.0], [0.0, 0.0]) == 0.0
            w = similar([1.0, 2.0])
            apply_inertial!(w, r, 3, [1.0, 2.0], [10.0, 20.0])
            @test w == [1.0, 2.0]
        end

        @testset "Inertial" begin
            r = Inertial(0.25)
            @test r.θ == 0.25
            @test inertial_coef(r, 0, [1.0], [0.0]) == 0.0
            @test inertial_coef(r, 5, [1.0], [1.0]) == 0.25
            θk = inertial_coef(r, 10, [1.0, 0.0], [0.0, 0.0])
            @test θk == min(0.25, 1.0 / (100 * 1.0))

            @test_throws ArgumentError Inertial(0.0)
            @test_throws ArgumentError Inertial(1.0)
        end

        @testset "apply_inertial!" begin
            r = Inertial(0.25)
            xk   = [1.0, 2.0]
            xkm1 = [0.0, 0.0]
            w = similar(xk)
            w_, θ = apply_inertial!(w, r, 1, xk, xkm1)
            @test w_ === w
            @test w == xk .+ θ .* (xk .- xkm1)
        end
    end

    # ========================================================================
    # Search direction: SpectralThreeTerm
    # ========================================================================

    @testset "Search direction (SpectralThreeTerm)" begin
        rule = SpectralThreeTerm(; r=0.1, alpha_bar=1.0)
        n = 5

        # Helper to build a direction! context NamedTuple
        _ctx(Fw, Fw_prev, w, w_prev, d_prev, k; α_prev=1.0) =
            (; Fw=Fw, Fw_prev=Fw_prev, w=w, w_prev=w_prev, d_prev=d_prev, k=k, α_prev=α_prev)

        @testset "k = 0: d = -F(w)" begin
            Fw = randn(MersenneTwister(1), n)
            d  = similar(Fw)
            direction!(d, rule, _ctx(Fw, Fw, Fw, Fw, similar(Fw), 0))
            @test d ≈ -Fw
        end

        @testset "k ≥ 1: finite output" begin
            rng = MersenneTwister(42)
            w       = randn(rng, n)
            w_prev  = randn(rng, n)
            d_prev  = randn(rng, n)
            Fw      = randn(rng, n)
            Fw_prev = randn(rng, n)
            d       = similar(w)

            direction!(d, rule, _ctx(Fw, Fw_prev, w, w_prev, d_prev, 1))
            @test all(isfinite, d)
        end

        @testset "constructor: alpha_min / alpha_max defaults + overrides" begin
            r_default = SpectralThreeTerm()
            @test r_default.alpha_min == 1e-10
            @test r_default.alpha_max == 1e30

            r_custom = SpectralThreeTerm(; alpha_min = 0.5, alpha_max = 2.0)
            @test r_custom.alpha_min == 0.5
            @test r_custom.alpha_max == 2.0
        end

        @testset "ϑ_I clamp: lower bound active" begin
            # Force the raw spectral coefficient s'y/y'y to be very small
            # (≈ 0) by choosing y_{k-1} ≂̸ 0 with s ⟂ y. Set alpha_min = 0.5
            # so the clamp lifts ϑ_I from ~0 to 0.5. With Fw_prev = 0, β_k
            # and ϑ_II vanish, so d_k = -0.5 · F(w_k) exactly.
            rng = MersenneTwister(7)
            w       = randn(rng, n)
            y       = randn(rng, n)            # y_{k-1}
            Fw_prev = randn(rng, n)
            Fw      = Fw_prev .+ y             # so F(w_k) - F(w_{k-1}) = y
            # Choose w_prev so that w - w_prev = -r·y + e, with e ⟂ y; then
            # s = (w - w_prev) + r·y = e ⟂ y, hence s'y = 0.
            e = randn(rng, n)
            e .-= (dot(e, y) / dot(y, y)) .* y    # project e onto y^⊥
            r_val = 0.1
            w_prev = w .+ r_val .* y .- e
            d_prev = zeros(n)                  # so β_k and ϑ_II annihilate
            # Disable Fw_prev's effect by zeroing it (v_k stays positive via
            # alpha_bar branch with zero d_prev → fallback to eps guard).
            # We instead set Fw_prev so that ‖Fw_prev‖² > 0; β_k and ϑ_II
            # depend on F_w · y and F_w · d_prev. With d_prev = 0, ϑ_II = 0.
            # β_k is generally nonzero unless F_w ⟂ y. Force F_w ⟂ y:
            Fw .-= (dot(Fw, y) / dot(y, y)) .* y    # now F_w ⟂ y → β_k = 0
            Fw_prev .= Fw .- y                      # preserve y = F_w - F_w_prev

            rule_lo = SpectralThreeTerm(; r = r_val, alpha_min = 0.5, alpha_max = 1e30)
            d = similar(w)
            direction!(d, rule_lo, _ctx(Fw, Fw_prev, w, w_prev, d_prev, 1))
            @test d ≈ -0.5 .* Fw atol = 1e-10
        end

        @testset "ϑ_I clamp: upper bound active" begin
            # Mirror of the previous test, but force s'y/y'y to be very
            # large by making y small and aligning s with y. Then alpha_max
            # caps ϑ_I.
            rng = MersenneTwister(13)
            w       = randn(rng, n)
            y       = 1e-6 .* randn(rng, n)        # tiny y → ‖y‖² very small
            Fw_prev = randn(rng, n)
            # F_w must satisfy F_w - F_w_prev = y AND F_w ⟂ y.
            Fw = Fw_prev .+ y
            Fw .-= (dot(Fw, y) / dot(y, y)) .* y    # F_w ⟂ y → β_k = 0
            Fw_prev .= Fw .- y
            r_val = 0.1
            # Pick s = (w - w_prev) + r·y aligned with y so s'y/y'y is huge.
            scale = 1e6
            s = scale .* y                          # s ∥ y, s'y/y'y = scale
            w_prev = w .+ r_val .* y .- s
            d_prev = zeros(n)

            rule_hi = SpectralThreeTerm(; r = r_val, alpha_min = 1e-10, alpha_max = 2.0)
            d = similar(w)
            direction!(d, rule_hi, _ctx(Fw, Fw_prev, w, w_prev, d_prev, 1))
            @test d ≈ -2.0 .* Fw atol = 1e-8
        end

        @testset "degenerate y = 0: ϑ_I falls back to alpha_min" begin
            # When y_{k-1} = 0 (so yy_sq = 0), ϑ_I should fall back to
            # alpha_min, NOT to zero — the pre-fix `zero(T)` fallback would
            # have produced d = β·d_prev (no descent contribution from F).
            # With Fw_prev = Fw the y vector is zero. v_k stays positive
            # via Fwm_sq, so β_k = (F_w · 0)/v_k = 0 and ϑ_II = (F_w·d_prev)/v_k.
            # Then d = -alpha_min · F_w + 0 - ϑ_II · 0 = -alpha_min · F_w
            # provided d_prev ⟂ F_w; we force that.
            n_local = 4
            Fw      = [1.0, 0.0, 1.0, 0.0]
            Fw_prev = copy(Fw)                  # → y = 0
            w       = randn(MersenneTwister(99), n_local)
            w_prev  = randn(MersenneTwister(100), n_local)
            d_prev  = [0.0, 1.0, 0.0, 1.0]      # ⟂ Fw

            rule_deg = SpectralThreeTerm(; alpha_min = 0.25, alpha_max = 1e30)
            d = similar(Fw)
            direction!(d, rule_deg, _ctx(Fw, Fw_prev, w, w_prev, d_prev, 1))
            @test d ≈ -0.25 .* Fw atol = 1e-10
        end

        @testset "default alpha_min/alpha_max do not perturb generic case" begin
            # Regression: with non-pathological inputs, sy/yy_sq lands in
            # [1e-10, 1e30] and the clamp is a no-op. Output matches the
            # pre-fix formula byte-equivalently.
            rng = MersenneTwister(2026)
            w       = randn(rng, n)
            w_prev  = randn(rng, n)
            d_prev  = randn(rng, n)
            Fw      = randn(rng, n)
            Fw_prev = randn(rng, n)

            # Recompute ϑ_I, β_k, ϑ_II, d the way the pre-fix code did
            # (no clamp) and confirm bitwise-equal output for default knobs.
            r_val = 0.1
            ab    = 1.0
            y     = Fw .- Fw_prev
            s     = (w .- w_prev) .+ r_val .* y
            yy_sq = dot(y, y)
            sy    = dot(s, y)
            Fw_y  = dot(Fw, y)
            Fw_d  = dot(Fw, d_prev)
            dd    = norm(d_prev)
            yyn   = sqrt(yy_sq)
            Fwm_sq = dot(Fw_prev, Fw_prev)
            v_k   = max(ab * dd * yyn, Fwm_sq)
            v_k   = max(v_k, eps(typeof(v_k)))
            ϑ_I_ref = sy / yy_sq               # raw, no clamp
            β_ref   = Fw_y / v_k
            ϑ_II_ref = Fw_d / v_k
            # Sanity: under random inputs the raw ϑ_I lands well inside [1e-10, 1e30]
            @test 1e-10 < abs(ϑ_I_ref) < 1e30
            d_ref = -ϑ_I_ref .* Fw .+ β_ref .* d_prev .- ϑ_II_ref .* y

            d = similar(w)
            direction!(d, SpectralThreeTerm(), _ctx(Fw, Fw_prev, w, w_prev, d_prev, 1))
            @test d ≈ d_ref atol = 1e-12
        end
    end

    # ========================================================================
    # Line searches
    # ========================================================================

    # ========================================================================
    # v0.2 SciML alignment (constraint moves from algorithm to problem)
    # ========================================================================

    @testset "v0.2 SciML alignment (Stage 6)" begin
        @testset "ConstrainedNonlinearProblem wraps a NonlinearProblem + set" begin
            f(u, p) = copy(u)
            inner = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            cprob = ConstrainedNonlinearProblem(inner, RealSpace())
            @test cprob.inner === inner
            @test cprob.set isa RealSpace
        end

        @testset "ConstrainedNonlinearProblem all-in-one constructor" begin
            f(u, p) = copy(u)
            cprob = ConstrainedNonlinearProblem(f, [1.0, 1.0]; set = HalfSpace([1.0, 1.0], 1.0))
            @test cprob.set isa HalfSpace
            @test cprob.inner.u0 == [1.0, 1.0]
        end

        @testset "_constraint_set resolves from prob.lb/ub or wrapper" begin
            f(u, p) = copy(u)
            # Unconstrained: returns RealSpace
            prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            @test DFMethods._constraint_set(prob) isa RealSpace

            # Box via lb/ub: returns BoxSet
            prob_box = SciMLBase.NonlinearProblem(f, [1.0, 1.0]; lb = -ones(2), ub = ones(2))
            set_box = DFMethods._constraint_set(prob_box)
            @test set_box isa BoxSet
            @test all(set_box.lower .== -1.0)
            @test all(set_box.upper .== 1.0)

            # ConstrainedNonlinearProblem: returns the wrapped set
            cprob = ConstrainedNonlinearProblem(f, [1.0, 1.0]; set = HalfSpace([1.0, 1.0], 1.0))
            @test DFMethods._constraint_set(cprob) isa HalfSpace
        end

        @testset "DFProjection has no `set` field" begin
            alg = DFProjection()
            @test !hasfield(typeof(alg), :set)
        end

        @testset "Box constraints via prob.lb/ub solve correctly" begin
            target = [0.3, -0.2]
            f(u, p) = u .- target
            prob = SciMLBase.NonlinearProblem(f, [1.0, -1.0]; lb = -ones(2), ub = ones(2))
            sol = solve(prob, DFProjection(inertial = NoInertial()))
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test sol.u ≈ target atol = 1e-4
        end

        @testset "ConstrainedNonlinearProblem with HalfSpace solves" begin
            f(u, p) = copy(u)
            cprob = ConstrainedNonlinearProblem(f, [1.0, 1.0]; set = HalfSpace([1.0, 1.0], 1.0))
            sol = solve(cprob, DFProjection(inertial = NoInertial()))
            @test sol.retcode == SciMLBase.ReturnCode.Success
        end

        @testset "One algorithm solves many problems" begin
            # The whole point: alg is problem-agnostic, can be reused.
            alg = DFProjection(direction = SpectralThreeTerm(),
                                linesearch = ResidualNormBacktrack(),
                                inertial = NoInertial())
            f(u, p) = copy(u)

            prob_unc = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            prob_box = SciMLBase.NonlinearProblem(f, [1.0, 1.0]; lb = -ones(2), ub = ones(2))
            cprob_hs = ConstrainedNonlinearProblem(f, [1.0, 1.0]; set = HalfSpace([1.0, 1.0], 0.5))

            sol1 = solve(prob_unc, alg)
            sol2 = solve(prob_box, alg)
            sol3 = solve(cprob_hs, alg)
            @test sol1.retcode == SciMLBase.ReturnCode.Success
            @test sol2.retcode == SciMLBase.ReturnCode.Success
            @test sol3.retcode == SciMLBase.ReturnCode.Success
        end

        @testset "Cache holds the resolved set" begin
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0]; lb = -ones(2), ub = ones(2))
            scml_cache = init(prob, DFProjection())
            inner = scml_cache.inner
            @test inner.set isa BoxSet
        end
    end

    # ========================================================================
    # v0.2 iterate update (Section A) — pluggable steps 6–7
    # ========================================================================

    @testset "v0.2 iterate update (Section A)" begin
        @testset "AbstractIterateUpdate hierarchy" begin
            @test SolodovSvaiterProjection() isa AbstractIterateUpdate
            @test DirectUpdate()             isa AbstractIterateUpdate
            @test HalpernUpdate(0.5)         isa AbstractIterateUpdate
        end

        @testset "Default DFProjection uses SolodovSvaiterProjection" begin
            alg = DFProjection()
            @test alg.iterate_update isa SolodovSvaiterProjection
        end

        @testset "SolodovSvaiterProjection: end-to-end solve" begin
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            alg = DFProjection(;
                iterate_update = SolodovSvaiterProjection(),
                linesearch     = ConstantBacktrack(),
                inertial       = NoInertial(),
                abstol         = 1e-6,
                maxiters       = 500,
            )
            sol = solve(prob, alg)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test norm(sol.u) <= 1e-5
        end

        @testset "SolodovSvaiterProjection: γ default + override" begin
            @test SolodovSvaiterProjection().γ == 1.0
            @test SolodovSvaiterProjection(; γ = 1.8).γ == 1.8
            @test SolodovSvaiterProjection(; γ = 0.5).γ == 0.5
        end

        @testset "SolodovSvaiterProjection: γ = 1 byte-equivalent to v0.3.1 default" begin
            # The default γ = 1.0 must reproduce v0.3.1's iterate stream.
            # Use a deterministic problem and compare two solves: one with
            # SolodovSvaiterProjection() (defaults), one with γ = 1.0 explicit.
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [0.7, -0.3, 0.5])
            alg_default  = DFProjection(; iterate_update = SolodovSvaiterProjection(),
                                            linesearch = ConstantBacktrack(),
                                            inertial   = NoInertial())
            alg_explicit = DFProjection(; iterate_update = SolodovSvaiterProjection(; γ = 1.0),
                                            linesearch = ConstantBacktrack(),
                                            inertial   = NoInertial())
            sol_d = solve(prob, alg_default)
            sol_e = solve(prob, alg_explicit)
            @test sol_d.u == sol_e.u
            @test sol_d.retcode == sol_e.retcode
        end

        @testset "SolodovSvaiterProjection: γ > 1 over-relaxation converges" begin
            # γ = 1.8 (STTDFPM 2024 experimental default) should still solve
            # cleanly on a strongly-monotone test problem.
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            alg = DFProjection(;
                iterate_update = SolodovSvaiterProjection(; γ = 1.8),
                linesearch     = ConstantBacktrack(),
                inertial       = NoInertial(),
                abstol         = 1e-6,
                maxiters       = 500,
            )
            sol = solve(prob, alg)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test norm(sol.u) <= 1e-5
        end

        @testset "SolodovSvaiterProjection: γ ≤ 1 cancels in RealSpace (documented)" begin
            # For X = RealSpace, the halfspace projection brings the target
            # w − γ·λ·F(z) back to the H_k boundary for γ ≤ 1, recovering
            # the γ = 1 iterate. Confirm by checking that solves with γ = 0.5
            # and γ = 1.0 produce the same iterate sequence.
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [0.8, -0.4])
            alg_half = DFProjection(; iterate_update = SolodovSvaiterProjection(; γ = 0.5),
                                       linesearch = ConstantBacktrack(),
                                       inertial   = NoInertial())
            alg_one  = DFProjection(; iterate_update = SolodovSvaiterProjection(; γ = 1.0),
                                       linesearch = ConstantBacktrack(),
                                       inertial   = NoInertial())
            sol_h = solve(prob, alg_half)
            sol_o = solve(prob, alg_one)
            @test sol_h.u ≈ sol_o.u  atol = 1e-12
            @test sol_h.stats.nsteps == sol_o.stats.nsteps
        end

        @testset "SolodovSvaiterProjection: γ > 1 with BoxSet still converges" begin
            # Box-constrained smoke: γ > 1 with a non-trivial X. The H_k
            # cancellation no longer applies (joint projection onto X ∩ H_k
            # is shaped by both constraints), so γ has a genuine effect.
            target = [0.3, -0.2]
            f(u, p) = u .- target
            prob = SciMLBase.NonlinearProblem(f, [1.0, -1.0]; lb = -ones(2), ub = ones(2))
            alg = DFProjection(;
                iterate_update = SolodovSvaiterProjection(; γ = 1.6),
                inertial       = NoInertial(),
                abstol         = 1e-6,
                maxiters       = 500,
            )
            sol = solve(prob, alg)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test sol.u ≈ target atol = 1e-4
        end

        @testset "DirectUpdate: x_{k+1} = project(z, set)" begin
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            alg = DFProjection(;
                iterate_update = DirectUpdate(),
                linesearch     = ConstantBacktrack(),
                inertial       = NoInertial(),
                maxiters       = 500,
            )
            sol = solve(prob, alg)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test norm(sol.u) <= 1e-4
        end

        @testset "HalpernUpdate with constant β=0 (≈ DirectUpdate)" begin
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            alg = DFProjection(;
                iterate_update = HalpernUpdate(0.0),
                linesearch     = ConstantBacktrack(),
                inertial       = NoInertial(),
                maxiters       = 500,
            )
            sol = solve(prob, alg)
            @test sol.retcode == SciMLBase.ReturnCode.Success
        end

        @testset "HalpernUpdate with β = k -> 1/(k+2)" begin
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            alg = DFProjection(;
                iterate_update = HalpernUpdate(k -> 1.0 / (k + 2)),
                linesearch     = ConstantBacktrack(),
                inertial       = NoInertial(),
                maxiters       = 1000,
            )
            sol = solve(prob, alg)
            @test SciMLBase.successful_retcode(sol.retcode) ||
                  sol.retcode == SciMLBase.ReturnCode.MaxIters
        end

        @testset "init_state for iterate-update strategies" begin
            f(u, p) = copy(u); x0 = [1.0, 1.0]; alg = DFProjection()
            prob = SciMLBase.NonlinearProblem(f, x0)
            @test DFMethods.init_state(SolodovSvaiterProjection(), prob, x0, alg) isa DFMethods.SolodovSvaiterState
            @test DFMethods.init_state(DirectUpdate(),             prob, x0, alg) === nothing
            @test DFMethods.init_state(HalpernUpdate(0.5),         prob, x0, alg) isa DFMethods.HalpernState
        end

        @testset "HalpernUpdate maintains feasibility on box-constrained problem" begin
            # Solution of F(u) = u is u* = 0, which is inside [-0.5, 0.5]^2.
            # The line-search trial points z_k can land outside the box; without
            # the final P_X step in HalpernUpdate, sol.u would generally violate
            # the bounds. With the projection in place, every iterate (including
            # the final one) must lie in the box.
            f(u, p) = copy(u)
            x0     = [0.4, 0.4]
            lb, ub = fill(-0.5, 2), fill(0.5, 2)
            prob   = SciMLBase.NonlinearProblem(f, x0; lb = lb, ub = ub)
            alg    = DFProjection(;
                iterate_update = HalpernUpdate(0.3),
                linesearch     = ConstantBacktrack(),
                inertial       = NoInertial(),
                maxiters       = 200,
            )
            sol = solve(prob, alg)
            @test all(lb .- 1e-12 .≤ sol.u .≤ ub .+ 1e-12)
        end
    end

    # ========================================================================
    # v0.2 line searches (Section B) — LineSearch.jl-aligned
    # ========================================================================

    @testset "v0.2 line searches (Section B)" begin
        @testset "Types are LineSearch.AbstractLineSearchAlgorithm subtypes" begin
            @test ConstantBacktrack()         isa LineSearch.AbstractLineSearchAlgorithm
            @test ResidualNormBacktrack()     isa LineSearch.AbstractLineSearchAlgorithm
            @test AdaptiveClampedBacktrack()  isa LineSearch.AbstractLineSearchAlgorithm
        end

        @testset "ConstantBacktrack init + solve!" begin
            f(u, p) = copy(u)
            u  = [1.0, 1.0]
            du = -f(u, nothing)            # descent direction
            prob = SciMLBase.NonlinearProblem(f, u)
            fu = f(u, nothing)
            ls = ConstantBacktrack()

            cache = CommonSolve.init(prob, ls, fu, u)
            @test cache isa LineSearch.AbstractLineSearchCache

            sol = CommonSolve.solve!(cache, u, du)
            @test sol isa LineSearch.LineSearchSolution
            @test 0 < sol.step_size <= 1
            @test sol.retcode == SciMLBase.ReturnCode.Success

            # Our caches expose rich state for the outer step!
            @test cache.n_evals >= 1
            @test all(isfinite, cache.fu_cache)
            @test all(isfinite, cache.z_cache)
        end

        @testset "ResidualNormBacktrack init + solve!" begin
            f(u, p) = copy(u)
            u  = [1.0, 1.0]
            du = -f(u, nothing)
            prob = SciMLBase.NonlinearProblem(f, u)
            fu = f(u, nothing)
            ls = ResidualNormBacktrack()

            cache = CommonSolve.init(prob, ls, fu, u)
            sol = CommonSolve.solve!(cache, u, du)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test 0 < sol.step_size <= 1
        end

        @testset "AdaptiveClampedBacktrack init + solve!" begin
            f(u, p) = copy(u)
            u  = [1.0, 1.0]
            du = -f(u, nothing)
            prob = SciMLBase.NonlinearProblem(f, u)
            fu = f(u, nothing)
            ls = AdaptiveClampedBacktrack(lo = 0.5, Δ_init = 4.0)

            cache = CommonSolve.init(prob, ls, fu, u)
            sol = CommonSolve.solve!(cache, u, du)
            @test sol.retcode == SciMLBase.ReturnCode.Success
        end

        @testset "Failure retcode when descent cannot be satisfied" begin
            # Construct du anti-aligned with -F(z) so backtracking can't satisfy
            # the Armijo condition within maxbt steps.
            f(u, p) = copy(u)
            u  = [1.0, 1.0]
            du = +f(u, nothing)            # NOT a descent direction
            prob = SciMLBase.NonlinearProblem(f, u)
            fu = f(u, nothing)
            ls = ConstantBacktrack(maxbt = 5)

            cache = CommonSolve.init(prob, ls, fu, u)
            sol = CommonSolve.solve!(cache, u, du)
            @test sol.retcode == SciMLBase.ReturnCode.Failure
        end
    end

    # ========================================================================
    # Approximate projection
    # ========================================================================

    @testset "approx_project_X_halfspace!" begin
        n = 2
        p_buf = zeros(n); q_buf = zeros(n)
        scratch = zeros(n); out_prev = zeros(n)
        out = zeros(n)

        @testset "RealSpace × halfspace specialization" begin
            # H = {x : x₁ + x₂ ≤ 1}, target outside
            target = [1.0, 1.0]
            approx_project_X_halfspace!(out, target, RealSpace(),
                                        [1.0, 1.0], 1.0, 1e-12,
                                        p_buf, q_buf, scratch, out_prev)
            @test sum(out) ≈ 1.0 atol=1e-12
            @test out ≈ [0.5, 0.5] atol=1e-12

            # Target already inside
            approx_project_X_halfspace!(out, [0.3, 0.3], RealSpace(),
                                        [1.0, 1.0], 1.0, 1e-12,
                                        p_buf, q_buf, scratch, out_prev)
            @test out ≈ [0.3, 0.3] atol=1e-12
        end

        @testset "BoxSet × halfspace (Dykstra)" begin
            # Box [-1, 1]² ∩ {x₁ + x₂ ≤ 1}, target outside both
            target = [2.0, 2.0]
            box = BoxSet([-1.0, -1.0], [1.0, 1.0])
            approx_project_X_halfspace!(out, target, box,
                                        [1.0, 1.0], 1.0, 1e-14,
                                        p_buf, q_buf, scratch, out_prev;
                                        maxiter = 500)
            @test all(-1.0 .<= out .<= 1.0)
            @test sum(out) <= 1.0 + 1e-6
            @test out ≈ [0.5, 0.5] atol=1e-4
        end
    end

    # ========================================================================
    # Algorithm: DFProjection via the SciML solve interface
    # ========================================================================

    @testset "DFProjection via solve" begin
        @testset "Unconstrained linear F: F(x) = x → x* = 0" begin
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            alg = DFProjection(;
                direction  = SpectralThreeTerm(),
                linesearch = ConstantBacktrack(),
                inertial   = NoInertial(),
                abstol     = 1e-6,
                maxiters   = 500,
            )
            sol = solve(prob, alg)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test norm(sol.u) <= 1e-5
            @test sol.stats.nsteps < 500
            @test sol.stats.nf > 0
        end

        @testset "Affine F with box (constraint via prob.lb/ub)" begin
            target = [0.3, -0.2]
            f(u, p) = u .- target
            prob = SciMLBase.NonlinearProblem(f, [1.0, -1.0]; lb = -ones(2), ub = ones(2))
            alg = DFProjection(;
                direction  = SpectralThreeTerm(),
                linesearch = ResidualNormBacktrack(),
                inertial   = Inertial(0.25),
                abstol     = 1e-6,
                maxiters   = 1000,
            )
            sol = solve(prob, alg)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test sol.u ≈ target atol=1e-4
        end

        @testset "Infeasible x0 gets projected (init_cache with explicit set)" begin
            F = x -> copy(x)
            x0_infeas = [5.0, 5.0]      # outside [-1, 1]²
            alg = DFProjection(; abstol = 1e-6, maxiters = 500)
            cache = init_cache(F, x0_infeas, alg; set = BoxSet([-1.0, -1.0], [1.0, 1.0]))
            @test all(-1.0 .<= cache.x .<= 1.0)     # init projected x0 onto box
        end

        @testset "Constructor defaults" begin
            alg = DFProjection()
            @test alg.direction  isa SpectralThreeTerm
            @test alg.linesearch isa ResidualNormBacktrack
            @test alg.inertial   isa Inertial
            @test alg.abstol     == 1e-6
            @test alg.maxiters   == 2000
            @test alg.ζ          == 0.5
        end

        @testset "v0.2 cache shape (Section D)" begin
            F = x -> copy(x)
            x0 = [1.0, 1.0]
            alg = DFProjection()
            cache = init_cache(F, x0, alg)

            # Four pluggable component-state slots
            @test hasfield(typeof(cache), :direction_state)
            @test hasfield(typeof(cache), :line_search_cache)
            @test hasfield(typeof(cache), :iterate_update_state)
            @test hasfield(typeof(cache), :stopping_state)

            # Stage 3c state: direction + stopping default to nothing;
            # line_search_cache holds a LineSearch.AbstractLineSearchCache
            # (the default `ResidualNormBacktrack`'s cache);
            # iterate_update_state holds the projection scratch
            @test cache.direction_state === nothing
            @test cache.line_search_cache isa LineSearch.AbstractLineSearchCache
            @test cache.line_search_cache isa DFMethods.ResidualNormBacktrackCache
            @test cache.stopping_state === nothing
            @test cache.iterate_update_state isa DFMethods.SolodovSvaiterState
            @test length(cache.iterate_update_state.proj_target) == length(x0)

            # Old direct fields gone
            @test !hasfield(typeof(cache), :proj_target)
            @test !hasfield(typeof(cache), :proj_p)
            @test !hasfield(typeof(cache), :proj_q)
            @test !hasfield(typeof(cache), :proj_scratch)
            @test !hasfield(typeof(cache), :proj_out_prev)
        end

        @testset "init_state defaults to nothing" begin
            f(u, p) = copy(u)
            x0 = [1.0]; alg = DFProjection()
            prob = SciMLBase.NonlinearProblem(f, x0)
            @test DFMethods.init_state(SpectralThreeTerm(), prob, x0, alg) === nothing
            @test DFMethods.init_state(MaxIters(100), prob, x0, alg) === nothing
        end

        @testset "Base.show DFProjectionCache prints a one-liner" begin
            F = x -> copy(x); x0 = [1.0]; alg = DFProjection()
            cache = init_cache(F, x0, alg)
            s = sprint(show, cache)
            @test occursin("DFProjectionCache", s)
            # The long parametric type signature shouldn't dominate
            @test length(s) < 200
        end

        @testset "α_prev plumbing through ctx" begin
            # Uses _RecorderDir + _RECORDED_ALPHA_PREV defined at top of file.
            # Exercises the full SciML path: NonlinearProblem → init → step!.
            _RECORDED_ALPHA_PREV[] = NaN

            F(u, p) = u .- sin.(u)
            prob = SciMLBase.NonlinearProblem(F, ones(10))
            alg = DFProjection(;
                direction = _RecorderDir(),
                inertial  = NoInertial(),
                maxiters  = 5,
            )
            cache = init(prob, alg)        # CommonSolve.init → DFSciMLCache

            # k=0 — α_prev should be the initial value (1.0)
            step!(cache)                   # CommonSolve.step!(::DFSciMLCache)
            @test _RECORDED_ALPHA_PREV[] == 1.0

            # k=1 — α_prev should be the α from k=0's line search (∈ (0, 1])
            step!(cache)
            @test 0 < _RECORDED_ALPHA_PREV[] <= 1.0
        end

        @testset "direction_state surfaced through ctx (Phase 2 regression)" begin
            # Uses _CountingDir + _CountingState defined at top of file.
            # init_state allocates state on cache.direction_state; the
            # framework passes it as ctx.direction_state to direction!;
            # mutations there are visible back on the cache. Would have
            # failed on v0.2 (no `direction_state` in direction!'s ctx).
            F(u, p) = u .- sin.(u)
            prob = SciMLBase.NonlinearProblem(F, ones(10))
            alg = DFProjection(;
                direction = _CountingDir(),
                inertial  = NoInertial(),
                maxiters  = 5,
            )
            cache = init(prob, alg)

            @test cache.inner.direction_state isa _CountingState
            @test cache.inner.direction_state.calls == 0

            step!(cache)
            @test cache.inner.direction_state.calls == 1

            step!(cache)
            @test cache.inner.direction_state.calls == 2
        end

        @testset "ctx schema pin — direction! (catches doc drift)" begin
            # If this assertion fails, the direction! ctx schema has changed.
            # Update the docs FIRST, then update the expected tuple here:
            #   - docs/src/extending.md §0 *direction! ctx — N fields* table
            #   - docs/src/extending.md §2 *Search direction* ctx table
            _CAPTURED_DIR_CTX[] = ()
            F(u, p) = u .- sin.(u)
            prob = SciMLBase.NonlinearProblem(F, ones(5))
            alg = DFProjection(;
                direction = _DirCtxCapture(),
                inertial  = NoInertial(),
                maxiters  = 1,
            )
            cache = init(prob, alg)
            step!(cache)
            @test _CAPTURED_DIR_CTX[] === (:Fw, :Fw_prev, :w, :w_prev, :d_prev, :k, :α_prev, :direction_state)
        end

        @testset "ctx schema pin — update_iterate! (catches doc drift)" begin
            # If this assertion fails, the update_iterate! ctx schema has
            # changed. Update the docs FIRST, then the expected tuple here:
            #   - docs/src/extending.md §0 *update_iterate! ctx — N fields*
            #   - docs/src/extending.md §3 *Iterate update* ctx description
            _CAPTURED_IU_CTX[] = ()
            F(u, p) = u .- sin.(u)
            prob = SciMLBase.NonlinearProblem(F, ones(5))
            alg = DFProjection(;
                iterate_update = _IUCtxCapture(),
                inertial       = NoInertial(),
                maxiters       = 1,
            )
            cache = init(prob, alg)
            step!(cache)
            @test _CAPTURED_IU_CTX[] === (:w, :d, :α, :z, :Fw, :Fz, :set, :k, :ζ, :inner_maxiter, :state)
        end

        @testset "NonlinearSolution fields from solve" begin
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [0.5, 0.5])
            sol = solve(prob, DFProjection(; maxiters=100))
            @test sol isa SciMLBase.AbstractNonlinearSolution
            @test sol.u isa Vector{<:AbstractFloat}
            @test sol.resid isa Vector{<:AbstractFloat}
            @test sol.stats.nsteps >= 0
            @test sol.stats.nf    >= 1
            @test sol.retcode    isa SciMLBase.ReturnCode.T
        end
    end

    # ========================================================================
    # v0.2 callbacks (Section C)
    # ========================================================================

    @testset "v0.2 callbacks (Section C)" begin
        @testset "AbstractCallback / AbstractStoppingCriterion hierarchy" begin
            @test HistoryCallback() isa AbstractCallback
            @test LoggingCallback(io = devnull) isa AbstractCallback
            @test AbsResidualTol(1e-6) isa AbstractCallback   # stopping criterion is a callback
            @test AbsResidualTol(1e-6) isa AbstractStoppingCriterion
        end

        @testset "HistoryCallback validates field names" begin
            @test_throws ErrorException HistoryCallback(fields = (:not_a_field,))
        end

        @testset "HistoryCallback rejects both fields and extractor" begin
            @test_throws ErrorException HistoryCallback(
                fields = (:k, :F_norm),
                extractor = c -> (a = 1,),
            )
        end

        @testset "HistoryCallback accumulates rows on :post_iter" begin
            hist = HistoryCallback(fields = (:k, :F_norm))
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            alg = DFProjection(;
                callbacks = AbstractCallback[hist],
                linesearch = ConstantBacktrack(),
                inertial = NoInertial(),
                maxiters = 10,
            )
            solve(prob, alg)
            @test length(hist.history) >= 1
            for row in hist.history
                @test row.k isa Integer
                @test row.F_norm isa AbstractFloat
            end
        end

        @testset "HistoryCallback with custom extractor" begin
            hist = HistoryCallback(extractor = c -> (custom = c.k * 2,))
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            alg = DFProjection(;
                callbacks = AbstractCallback[hist],
                linesearch = ConstantBacktrack(),
                inertial = NoInertial(),
                maxiters = 5,
            )
            solve(prob, alg)
            @test length(hist.history) >= 1
            for row in hist.history
                @test row.custom isa Integer
            end
        end

        @testset "LoggingCallback prints to io" begin
            io = IOBuffer()
            logger = LoggingCallback(io = io, columns = (:k, :F_norm), every = 1, footer = true)
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            alg = DFProjection(;
                callbacks = AbstractCallback[logger],
                linesearch = ConstantBacktrack(),
                inertial = NoInertial(),
                maxiters = 10,
            )
            solve(prob, alg)
            output = String(take!(io))
            @test occursin("k", output)         # header
            @test occursin("F_norm", output)
            @test occursin("Terminated", output)  # footer
        end

        @testset "LoggingCallback honors `every`" begin
            io = IOBuffer()
            logger = LoggingCallback(io = io, columns = (:k,), every = 100, footer = false)
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            alg = DFProjection(;
                callbacks = AbstractCallback[logger],
                linesearch = ConstantBacktrack(),
                inertial = NoInertial(),
                maxiters = 5,
            )
            solve(prob, alg)
            output = String(take!(io))
            # With every=100 and only ~5 iters, we should see the header but no rows
            @test occursin("k", output)
        end

        @testset "HISTORY_FIELDS constant" begin
            @test :k in HISTORY_FIELDS
            @test :F_norm in HISTORY_FIELDS
            @test :elapsed in HISTORY_FIELDS
        end
    end

    # ========================================================================
    # Stopping criteria
    # ========================================================================

    @testset "Stopping criteria" begin
        # Build a real cache via init_cache so we can probe criteria via on_event!
        F(x) = copy(x)              # F(x) = x, F(0) = 0
        x0 = [1.0, 1.0]
        alg_default = DFProjection()
        cache = init_cache(F, x0, alg_default)

        @testset "AbsResidualTol fires at :post_linesearch" begin
            c = AbsResidualTol(1e-6)

            cache.Fz .= [0.5, 0.5];   @test on_event!(c, cache, :post_linesearch) == (false, :Default)
            cache.Fz .= [1e-8, 1e-8]; @test on_event!(c, cache, :post_linesearch) == (true,  :Success)

            # Other events fall through
            @test on_event!(c, cache, :post_iter)    == (false, :Default)
            @test on_event!(c, cache, :initialize)   == (false, :Default)
            @test on_event!(c, cache, :terminate)    == (false, :Default)
        end

        @testset "RelResidualTol uses F0_norm at :post_linesearch" begin
            cache.F0_norm = 2.0
            c = RelResidualTol(1e-4)
            # threshold = 0 + 1e-4 * 2 = 2e-4
            cache.Fz .= [1e-3, 1e-3];     @test on_event!(c, cache, :post_linesearch) == (false, :Default)
            cache.Fz .= [1e-5, 1e-5];     @test on_event!(c, cache, :post_linesearch) == (true,  :Success)

            # with abstol kwarg
            c2 = RelResidualTol(1e-4; abstol = 1e-3)
            cache.Fz .= [5e-4, 5e-4];     @test on_event!(c2, cache, :post_linesearch) == (true, :Success)
        end

        @testset "StepNormTol fires at :post_iter only" begin
            c = StepNormTol(1e-8)

            cache.k = 0
            @test on_event!(c, cache, :post_iter) == (false, :Default)

            cache.k = 5
            cache.x .= [1.0, 1.0]; cache.x_prev .= [1.0, 1.0]
            @test on_event!(c, cache, :post_iter) == (true, :Stalled)

            cache.x_prev .= [0.0, 0.0]
            @test on_event!(c, cache, :post_iter) == (false, :Default)

            @test on_event!(c, cache, :post_linesearch) == (false, :Default)
        end

        @testset "DirectionNormTol fires at :post_iter only" begin
            c = DirectionNormTol(1e-10)

            cache.k = 0
            @test on_event!(c, cache, :post_iter) == (false, :Default)

            cache.k = 5
            cache.d .= [1e-12, 1e-12]
            @test on_event!(c, cache, :post_iter) == (true, :Stalled)

            cache.d .= [1.0, 1.0]
            @test on_event!(c, cache, :post_iter) == (false, :Default)
        end

        @testset "MaxIters / MaxFEvals / MaxTime fire at :post_iter" begin
            cache.k = 999;        @test on_event!(MaxIters(1000), cache, :post_iter) == (false, :Default)
            cache.k = 1000;       @test on_event!(MaxIters(1000), cache, :post_iter) == (true,  :MaxIters)

            cache.n_evals = 50;   @test on_event!(MaxFEvals(100), cache, :post_iter) == (false, :Default)
            cache.n_evals = 100;  @test on_event!(MaxFEvals(100), cache, :post_iter) == (true,  :MaxFEvals)

            cache.t_start = time() + 100.0   # in the future → no time elapsed
            @test on_event!(MaxTime(0.001), cache, :post_iter) == (false, :Default)
            cache.t_start = time() - 100.0   # 100 sec in the past
            @test on_event!(MaxTime(0.001), cache, :post_iter) == (true,  :MaxTime)
        end

        @testset "UserStop callback" begin
            c1 = UserStop(_cache -> (true, :CustomCode))
            stopped, code = on_event!(c1, cache, :post_iter)
            @test stopped
            @test code == :CustomCode

            c2 = UserStop(_cache -> (false, :Default))
            @test on_event!(c2, cache, :post_iter) == (false, :Default)
        end

        @testset "AnyOf composes (first to fire wins)" begin
            c = AnyOf(
                AbsResidualTol(1e-6),
                MaxIters(100),
            )
            # Neither fires
            cache.Fz .= [0.5, 0.5]
            cache.k = 50
            @test on_event!(c, cache, :post_linesearch) == (false, :Default)
            @test on_event!(c, cache, :post_iter)       == (false, :Default)

            # AbsResidualTol fires at :post_linesearch
            cache.Fz .= [1e-8, 1e-8]
            @test on_event!(c, cache, :post_linesearch) == (true, :Success)

            # MaxIters fires at :post_iter
            cache.k = 200
            cache.Fz .= [0.5, 0.5]
            @test on_event!(c, cache, :post_iter) == (true, :MaxIters)
        end

        @testset "DFProjection default stopping" begin
            alg = DFProjection()                       # abstol=1e-6, maxiters=2000 by default
            @test alg.stopping isa AnyOf
            @test alg.stopping.criteria[1] isa AbsResidualTol
            @test alg.stopping.criteria[1].abstol == 1e-6
            @test alg.stopping.criteria[2] isa MaxIters
            @test alg.stopping.criteria[2].maxiters == 2000

            alg2 = DFProjection(; abstol = 1e-10, maxiters = 5000)
            @test alg2.stopping.criteria[1].abstol == 1e-10
            @test alg2.stopping.criteria[2].maxiters == 5000
        end

        @testset "DFProjection custom stopping wins" begin
            custom = AnyOf(RelResidualTol(1e-8), MaxTime(60.0))
            alg = DFProjection(; stopping = custom, abstol = 1e-3, maxiters = 10)
            @test alg.stopping === custom              # custom preserved
            @test alg.abstol   == 1e-3                 # fields still stored
            @test alg.maxiters == 10                   # but unused by step!
        end

        @testset "SciML common-solver keyword routing" begin
            # Authoritative option list:
            # https://docs.sciml.ai/NonlinearSolve/stable/basics/solve/

            @testset "constructor reltol/maxtime build the default stopping" begin
                a = DFProjection(; reltol = 1e-3)
                @test a.reltol == 1e-3
                @test a.auto_stopping
                @test any(c isa RelResidualTol && c.rtol == 1e-3 for c in a.stopping.criteria)

                b = DFProjection(; maxtime = 5.0)
                @test b.maxtime == 5.0
                @test any(c isa MaxTime && c.maxtime == 5.0 for c in b.stopping.criteria)

                # Default omits both → stays AnyOf(AbsResidualTol, MaxIters).
                d = DFProjection()
                @test d.reltol == 0.0
                @test d.maxtime == Inf
                @test !any(c isa RelResidualTol for c in d.stopping.criteria)
                @test !any(c isa MaxTime       for c in d.stopping.criteria)
            end

            @testset "solve keywords rebuild the effective stopping" begin
                f(u, p) = copy(u)
                prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
                base = DFProjection()

                c1 = init(prob, base; reltol = 1e-3)
                @test c1.inner.alg.reltol == 1e-3
                @test any(c isa RelResidualTol && c.rtol == 1e-3
                          for c in c1.inner.alg.stopping.criteria)

                c2 = init(prob, base; maxtime = 2.0)
                @test c2.inner.alg.maxtime == 2.0
                @test any(c isa MaxTime && c.maxtime == 2.0
                          for c in c2.inner.alg.stopping.criteria)

                c3 = init(prob, base; abstol = 1e-9, maxiters = 321)
                @test c3.inner.alg.abstol   == 1e-9
                @test c3.inner.alg.maxiters == 321

                # No tol kwargs → no rebuild (same alg object flows through).
                @test init(prob, base).inner.alg === base
            end

            @testset "reltol drives convergence (behavioral)" begin
                F2(u, p) = u .- p
                x0 = [1.0, -1.0]; target = [0.3, -0.2]
                prob = SciMLBase.NonlinearProblem(F2, x0, target)
                alg  = DFProjection(; inertial = NoInertial())
                # Absurdly tight abstol so only the relative test can stop us.
                sol = solve(prob, alg; abstol = 1e-14, reltol = 1e-2, maxiters = 2000)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test norm(sol.resid) <= 1e-2 * norm(x0 .- target) + 1e-12
            end

            @testset "maxtime stops the solve (behavioral)" begin
                F4(u, p) = u .- sin.(u) .- 1.0
                prob = SciMLBase.NonlinearProblem(F4, ones(20))
                alg  = DFProjection(; inertial = NoInertial())
                # Zero time budget → first :post_iter MaxTime check fires.
                sol = solve(prob, alg; maxtime = 0.0, abstol = 1e-12, verbose = false)
                @test sol.retcode == SciMLBase.ReturnCode.Terminated
            end

            @testset "all standard SciML kwargs absorbed without error" begin
                f(u, p) = copy(u)
                prob = SciMLBase.NonlinearProblem(f, [0.5, 0.5])
                sol = solve(prob, DFProjection(; maxiters = 200);
                            termination_condition = nothing,
                            internalnorm          = norm,
                            alias_u0              = false,
                            show_trace            = Val(false),
                            store_trace           = Val(false),
                            trace_level           = nothing)
                @test sol isa SciMLBase.AbstractNonlinearSolution
            end

            @testset "verbose toggles the non-convergence warning" begin
                # Nonlinear F (not solved in a single projection step) so
                # maxiters=1 yields a genuine non-Success MaxIters exit.
                F5(u, p) = u .- sin.(u) .- 1.0
                prob = SciMLBase.NonlinearProblem(F5, ones(8))
                alg  = DFProjection(; maxiters = 1)

                @test init(prob, alg; verbose = false).verbose == false
                @test init(prob, alg).verbose == true

                # Default verbose=true → a non-Success exit warns.
                sol = @test_logs (:warn,) match_mode = :any solve(prob, alg)
                @test sol.retcode != SciMLBase.ReturnCode.Success
                # verbose=false → no warn-level log.
                @test_logs match_mode = :all min_level = Base.CoreLogging.Warn solve(prob, alg; verbose = false)
            end

            @testset "custom stopping + tol override warns, preserves rule" begin
                f(u, p) = copy(u)
                prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
                custom = AnyOf(AbsResidualTol(1e-4), MaxIters(50))
                alg = DFProjection(; stopping = custom)
                @test !alg.auto_stopping

                c = @test_logs (:warn,) match_mode = :any init(prob, alg; reltol = 1e-3)
                @test c.inner.alg.stopping === custom    # custom rule preserved
                # verbose=false silences the conflict warning.
                @test_logs match_mode = :all min_level = Base.CoreLogging.Warn init(prob, alg; reltol = 1e-3, verbose = false)
            end
        end

        @testset "end-to-end: RelResidualTol converges" begin
            F2(u, p) = u .- p
            target = [0.3, -0.2]
            prob = SciMLBase.NonlinearProblem(F2, [1.0, -1.0], target;
                                              lb = -ones(2), ub = ones(2))
            alg = DFProjection(;
                inertial = NoInertial(),
                stopping = AnyOf(RelResidualTol(1e-6; abstol = 1e-12), MaxIters(1000)),
            )
            sol = solve(prob, alg)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test sol.u ≈ target atol = 1e-4
        end

        @testset "end-to-end: StepNormTol triggers :Stalled" begin
            # A problem MPRPL-style stalls on — use a degenerate-Jacobian one.
            # Here we just force it via tight xtol and loose abstol.
            F3(u, p) = u .- sin.(u)
            prob = SciMLBase.NonlinearProblem(F3, ones(10))
            alg = DFProjection(;
                inertial = NoInertial(),
                # Very tight step tol catches us before residual tol.
                stopping = AnyOf(StepNormTol(1.0), MaxIters(2000)),
            )
            sol = solve(prob, alg)
            @test sol.retcode == SciMLBase.ReturnCode.Stalled
        end
    end

    # ========================================================================
    # SciMLBase integration (NonlinearProblem / solve / NonlinearSolution)
    # ========================================================================

    @testset "SciMLBase integration" begin

        @testset "DFProjection <: AbstractNonlinearAlgorithm" begin
            @test DFProjection() isa SciMLBase.AbstractNonlinearAlgorithm
        end

        @testset "Out-of-place NonlinearProblem: F(u,p) = u" begin
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [1.0, 1.0])
            sol = solve(prob, DFProjection(; inertial = NoInertial(),
                                            maxiters = 500))
            @test sol isa SciMLBase.AbstractNonlinearSolution
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test norm(sol.u) <= 1e-5
            @test sol.stats.nf >= 1
            @test sol.stats.nsteps >= 0
        end

        @testset "In-place NonlinearProblem: f!(du, u, p)" begin
            function f!(du, u, p)
                du .= u
                return nothing
            end
            prob = SciMLBase.NonlinearProblem(f!, [1.0, 1.0])
            sol = solve(prob, DFProjection(; inertial = NoInertial(),
                                            maxiters = 500))
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test norm(sol.u) <= 1e-5
        end

        @testset "Parameters p flow through: F(u,p) = u - p" begin
            f(u, p) = u .- p
            target = [0.3, -0.2]
            prob = SciMLBase.NonlinearProblem(f, [1.0, -1.0], target;
                                              lb = -ones(2), ub = ones(2))
            sol = solve(prob, DFProjection(;
                inertial = NoInertial(),
                maxiters = 1000,
            ))
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test sol.u ≈ target atol=1e-4
        end

        @testset "kwargs override abstol/maxiters" begin
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [1.0])
            sol = solve(prob, DFProjection(; inertial = NoInertial());
                        abstol = 1e-10, maxiters = 1000)
            @test abs(sol.u[1]) <= 1e-9
        end

        @testset "MaxIters retcode" begin
            # Need a problem that does NOT trivially converge at α=1.
            # For F(u)=u with d_0=-u, z_0 = w + α(-w) = 0 at α=1 — converges
            # in one step. Use Dai P2 form `u - sin(u)`: solution x* = 0,
            # but with the slow tail near zero (Jacobian = 1 - cos = 0 at 0)
            # so n=10 from ones(10) needs ~16 iterations. Forcing maxiters=2
            # guarantees we hit the cap.
            f(u, p) = u .- sin.(u)
            prob = SciMLBase.NonlinearProblem(f, ones(10))
            sol = solve(prob, DFProjection(; inertial = NoInertial(),
                                            abstol = 1e-12);
                        maxiters = 2)
            @test sol.retcode == SciMLBase.ReturnCode.MaxIters
        end

        @testset "NonlinearSolution shape" begin
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [0.5])
            sol = solve(prob, DFProjection(; maxiters = 100))
            @test sol.u isa Vector{<:AbstractFloat}
            @test sol.resid isa Vector{<:AbstractFloat}
            @test length(sol.resid) == length(sol.u)
            @test sol.alg isa DFProjection
        end

    end  # SciMLBase integration

    # ========================================================================
    # Element-type genericity (v0.3.0)
    # ========================================================================

    @testset "Element-type genericity (v0.3.0)" begin
        # Test problem: F(u) = u - sin(u), root at u = 0.
        F_t(u, p) = u .- sin.(u)

        @testset "Float32 — default config end-to-end" begin
            prob = SciMLBase.NonlinearProblem(F_t, Float32.(ones(10)))
            sol = solve(prob, DFProjection(); abstol = 1f-5, maxiters = 100)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test eltype(sol.u)     === Float32
            @test eltype(sol.resid) === Float32
            # F(u) = u - sin(u) has a cubic-residual root; test the residual
            # convergence criterion, not iterate distance from u=0.
            @test norm(sol.resid) <= 1f-5
        end

        @testset "Float32 — alternate components (DirectUpdate + ConstantBacktrack + NoInertial)" begin
            prob = SciMLBase.NonlinearProblem(F_t, Float32.(ones(5)))
            alg = DFProjection(;
                iterate_update = DirectUpdate(),
                linesearch     = ConstantBacktrack(σ = 0.01, ρ = 0.5),
                inertial       = NoInertial(),
                abstol         = 1f-4,
                maxiters       = 500,
            )
            sol = solve(prob, alg)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test eltype(sol.u) === Float32
        end

        @testset "Float32 — box constraints" begin
            n = 5
            prob = SciMLBase.NonlinearProblem(F_t, Float32.(ones(n));
                                              lb = Float32.(fill(-2, n)),
                                              ub = Float32.(fill(2, n)))
            sol = solve(prob, DFProjection(); abstol = 1f-5, maxiters = 100)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test eltype(sol.u) === Float32
            @test all(-2.0f0 .<= sol.u .<= 2.0f0)
        end

        @testset "BigFloat — high-precision sanity (≤5 s budget)" begin
            # Per Q3 (a) design decision: keep BigFloat coverage minimal.
            # n = 3 keeps per-iteration cost low; tight abstol exercises
            # precision well below Float64's reach.
            prob = SciMLBase.NonlinearProblem(F_t, BigFloat.(ones(3)))
            sol = solve(prob, DFProjection(); abstol = BigFloat(1e-30),
                                              maxiters = 200)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test eltype(sol.u)     === BigFloat
            @test eltype(sol.resid) === BigFloat
            # Residual precision well below Float64's eps — a Float64 solve
            # could never achieve this regardless of iterate count.
            @test norm(sol.resid) < eps(Float64)
        end
    end

end  # @testset "DFMethods.jl"
