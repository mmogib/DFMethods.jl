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

        @testset "CappedBox (Ibrahim 2026 Ω)" begin
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
    end

    # ========================================================================
    # Line searches
    # ========================================================================

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
                set        = RealSpace(),
                abstol     = 1e-6,
                maxiters   = 500,
            )
            sol = solve(prob, alg)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test norm(sol.u) <= 1e-5
            @test sol.stats.nsteps < 500
            @test sol.stats.nf > 0
        end

        @testset "Affine F with box: F(x) = x - target, target inside box" begin
            target = [0.3, -0.2]
            f(u, p) = u .- target
            prob = SciMLBase.NonlinearProblem(f, [1.0, -1.0])
            alg = DFProjection(;
                direction  = SpectralThreeTerm(),
                linesearch = ResidualNormBacktrack(),
                inertial   = Inertial(0.25),
                set        = BoxSet([-1.0, -1.0], [1.0, 1.0]),
                abstol     = 1e-6,
                maxiters   = 1000,
            )
            sol = solve(prob, alg)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test sol.u ≈ target atol=1e-4
        end

        @testset "Infeasible x0 gets projected" begin
            F = x -> copy(x)
            x0_infeas = [5.0, 5.0]      # outside [-1, 1]²
            alg = DFProjection(;
                set      = BoxSet([-1.0, -1.0], [1.0, 1.0]),
                abstol   = 1e-6,
                maxiters = 500,
            )
            cache = init_cache(F, x0_infeas, alg)
            @test all(-1.0 .<= cache.x .<= 1.0)     # init projected x0 onto box
        end

        @testset "Constructor defaults" begin
            alg = DFProjection()
            @test alg.direction  isa SpectralThreeTerm
            @test alg.linesearch isa ResidualNormBacktrack
            @test alg.inertial   isa Inertial
            @test alg.set        isa RealSpace
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

        @testset "NonlinearSolution fields from solve" begin
            f(u, p) = copy(u)
            prob = SciMLBase.NonlinearProblem(f, [0.5, 0.5])
            sol = solve(prob, DFProjection(; maxiters=100))
            @test sol isa SciMLBase.AbstractNonlinearSolution
            @test sol.u isa Vector{Float64}
            @test sol.resid isa Vector{Float64}
            @test sol.stats.nsteps >= 0
            @test sol.stats.nf    >= 1
            @test sol.retcode    isa SciMLBase.ReturnCode.T
        end
    end

    # ========================================================================
    # Stopping criteria
    # ========================================================================

    @testset "Stopping criteria" begin
        # Build a real cache via init_cache so we can probe criteria
        F(x) = copy(x)              # F(x) = x, F(0) = 0
        x0 = [1.0, 1.0]
        alg_default = DFProjection()
        cache = init_cache(F, x0, alg_default)

        @testset "AbsResidualTol fires at_w / at_z, not at_end" begin
            c = AbsResidualTol(1e-6)

            cache.Fw .= [0.5, 0.5];   @test should_stop_at_w(c, cache) == (false, :Default)
            cache.Fw .= [1e-8, 1e-8]; @test should_stop_at_w(c, cache) == (true,  :Success)

            cache.Fz .= [0.5, 0.5];   @test should_stop_at_z(c, cache) == (false, :Default)
            cache.Fz .= [1e-8, 1e-8]; @test should_stop_at_z(c, cache) == (true,  :Success)

            @test should_stop_at_end(c, cache) == (false, :Default)
        end

        @testset "RelResidualTol uses F0_norm" begin
            cache.F0_norm = 2.0
            c = RelResidualTol(1e-4)
            # threshold = 0 + 1e-4 * 2 = 2e-4
            cache.Fw .= [1e-3, 1e-3];     @test should_stop_at_w(c, cache) == (false, :Default)
            cache.Fw .= [1e-5, 1e-5];     @test should_stop_at_w(c, cache) == (true,  :Success)

            # with abstol kwarg
            c2 = RelResidualTol(1e-4; abstol = 1e-3)
            cache.Fw .= [5e-4, 5e-4];     @test should_stop_at_w(c2, cache) == (true, :Success)
        end

        @testset "StepNormTol fires at_end only" begin
            c = StepNormTol(1e-8)

            cache.k = 0   # no prev step yet
            @test should_stop_at_end(c, cache) == (false, :Default)

            cache.k = 5
            cache.x .= [1.0, 1.0]; cache.x_prev .= [1.0, 1.0]
            @test should_stop_at_end(c, cache) == (true, :Stalled)

            cache.x_prev .= [0.0, 0.0]
            @test should_stop_at_end(c, cache) == (false, :Default)

            @test should_stop_at_w(c, cache) == (false, :Default)
            @test should_stop_at_z(c, cache) == (false, :Default)
        end

        @testset "DirectionNormTol fires at_end only" begin
            c = DirectionNormTol(1e-10)

            cache.k = 0
            @test should_stop_at_end(c, cache) == (false, :Default)

            cache.k = 5
            cache.d .= [1e-12, 1e-12]
            @test should_stop_at_end(c, cache) == (true, :Stalled)

            cache.d .= [1.0, 1.0]
            @test should_stop_at_end(c, cache) == (false, :Default)
        end

        @testset "MaxIters / MaxFEvals / MaxTime" begin
            cache.k = 999;        @test should_stop_at_end(MaxIters(1000), cache) == (false, :Default)
            cache.k = 1000;       @test should_stop_at_end(MaxIters(1000), cache) == (true,  :MaxIters)

            cache.n_evals = 50;   @test should_stop_at_end(MaxFEvals(100), cache) == (false, :Default)
            cache.n_evals = 100;  @test should_stop_at_end(MaxFEvals(100), cache) == (true,  :MaxFEvals)

            cache.t_start = time() + 100.0   # in the future → no time elapsed
            @test should_stop_at_end(MaxTime(0.001), cache) == (false, :Default)
            cache.t_start = time() - 100.0   # 100 sec in the past
            @test should_stop_at_end(MaxTime(0.001), cache) == (true,  :MaxTime)
        end

        @testset "UserStop callback" begin
            c1 = UserStop(_cache -> (true, :CustomCode))
            stopped, code = should_stop_at_end(c1, cache)
            @test stopped
            @test code == :CustomCode

            c2 = UserStop(_cache -> (false, :Default))
            @test should_stop_at_end(c2, cache) == (false, :Default)
        end

        @testset "AnyOf composes (first to fire wins)" begin
            c = AnyOf(
                AbsResidualTol(1e-6),
                MaxIters(100),
            )
            # Neither fires
            cache.Fw .= [0.5, 0.5]
            cache.k = 50
            @test should_stop_at_w(c, cache)   == (false, :Default)
            @test should_stop_at_end(c, cache) == (false, :Default)

            # AbsResidualTol fires at_w
            cache.Fw .= [1e-8, 1e-8]
            @test should_stop_at_w(c, cache) == (true, :Success)

            # MaxIters fires at_end
            cache.k = 200
            @test should_stop_at_end(c, cache) == (true, :MaxIters)
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

        @testset "end-to-end: RelResidualTol converges" begin
            F2(u, p) = u .- p
            target = [0.3, -0.2]
            prob = SciMLBase.NonlinearProblem(F2, [1.0, -1.0], target)
            alg = DFProjection(;
                inertial = NoInertial(),
                set      = BoxSet([-1.0, -1.0], [1.0, 1.0]),
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
            prob = SciMLBase.NonlinearProblem(f, [1.0, -1.0], target)
            sol = solve(prob, DFProjection(;
                inertial = NoInertial(),
                set      = BoxSet([-1.0, -1.0], [1.0, 1.0]),
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
            @test sol.u isa Vector{Float64}
            @test sol.resid isa Vector{Float64}
            @test length(sol.resid) == length(sol.u)
            @test sol.alg isa DFProjection
        end

    end  # SciMLBase integration

end  # @testset "DFMethods.jl"
