# Changelog

All notable changes to **DFMethods.jl** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-20

Major restructuring toward **v0.2.0**. The release reorganizes `DFProjection`
into a framework with pluggable iterate-update strategies and callback-based
observation, adopts SciML-native conventions for constraint location and
line-search dispatch, and overhauls the documentation.

### Added
- `ConstrainedNonlinearProblem(inner::NonlinearProblem, set::AbstractConstraintSet)` for non-box constraints. Box constraints are now passed via the SciML-native `NonlinearProblem(f, u0; lb, ub)`.
- `AbstractIterateUpdate` family with three concrete strategies:
  - `SolodovSvaiterProjection()` (default) — the original v0.1 hyperplane projection.
  - `DirectUpdate()` — `x_{k+1} = P_X(z_k)`.
  - `HalpernUpdate(β)` — `x_{k+1} = P_X(β x_0 + (1 − β) z_k)` with scalar or schedule `β`.
  Each implements the `update_iterate!(x_new, rule, ctx)` contract.
- New line-search types subtyping `LineSearch.AbstractLineSearchAlgorithm`: `ConstantBacktrack`, `ResidualNormBacktrack` (default), `AdaptiveClampedBacktrack`. Any rule from the LineSearch.jl ecosystem now plugs in via the `CommonSolve.init` / `CommonSolve.solve!` contract.
- `AbstractCallback` hierarchy with `on_event!(cb, cache, event)` over a four-event lifecycle (`:initialize`, `:post_linesearch`, `:post_iter`, `:terminate`). `AbstractStoppingCriterion <: AbstractCallback`.
- Built-in observer callbacks: `HistoryCallback`, `LoggingCallback`; `HISTORY_FIELDS` constant.
- `init_state(component, prob, x0, alg)` contract for component-specific per-solve state.
- Documentation: new pages `references.md`, `benchmarks.md`, `comparisons.md`; `extending.md` rewritten with eight component subsections; algorithm and quickstart tone-aligned. `@example` blocks make code on narrative pages compile-checked; `jldoctest` blocks pin invariant behavior in constraint-set and inertial docstrings.
- Test suite: 127 → 223 tests.

### Changed (breaking)
- **Constraint location.** The feasibility set is now a property of the **problem**, not the algorithm. `DFProjection(; set = …)` is removed; pass constraints via `prob.lb`/`prob.ub` or `ConstrainedNonlinearProblem`. One configured algorithm can solve unconstrained, box, and arbitrary closed-convex problems unchanged.
- **`ψ` → `F` rename.** Cache field names (`ψw → Fw`, `ψw_prev → Fw_prev`, `ψz → Fz`, `ψ0_norm → F0_norm`), docstring math, and argument names switch from `\psi` to `F` throughout, matching SciML convention and the user-facing notation in `NonlinearProblem`.
- **`AbstractDFProjectionAlgorithm → AbstractDFProjection`** (renamed; drops the redundant `Algorithm` suffix).
- **Default line search renamed**: `LSII()` → `ResidualNormBacktrack()`. Other v0.1 line-search names (`LSI`–`LSVII`) are gone; the mapping is: `LSI → ConstantBacktrack`, `LSII → ResidualNormBacktrack`, `LSVII → AdaptiveClampedBacktrack`. `LSIII`, `LSIV`, `LSV`, `LSVI` are dropped (recoverable as user-defined line searches following the LineSearch.jl contract — see *Extending* §1).
- **Stopping criteria are now callbacks.** `should_stop_at_w`, `should_stop_at_z`, `should_stop_at_end` are replaced by `on_event!(crit, cache, event) → (Bool, Symbol)` over the four-event lifecycle.
- `DFProjection` constructor: new keyword arguments `iterate_update` and `callbacks`; removed `set`.

### Removed (breaking)
- `solve_df` / `solve_df!` / `DFSolution` — use the SciML interface `solve(prob, alg)` or the `init` / `step!` / `solve!` triplet.
- `AbstractDFLineSearch`, `LSI`, `LSII`, `LSIII`, `LSIV`, `LSV`, `LSVI`, `LSVII`, `gamma_k`, `linesearch!` — replaced by LineSearch.jl-aligned subtypes.
- Assumption-trait predicates `monotonicity_required`, `pseudomonotonicity_sufficient`, `convex_set_required` — never consulted at runtime; convergence assumptions are documented in prose on the Algorithm page.

### Fixed
- **`solve` dispatch under `using NonlinearSolve`.** Without an explicit `SciMLBase.__solve(::NonlinearProblem, ::DFProjection)` method, NonlinearSolveBase's generic `__solve(::NonlinearProblem, ::AbstractNonlinearAlgorithm)` intercepted the call and routed through its polyalgorithm path, breaking on box-constrained problems. The method is now registered; the `CommonSolve` init/solve! pipeline runs unchanged.
- **`HalpernUpdate` feasibility.** The previous implementation produced `x_{k+1} = β x_0 + (1 − β) z_k` without projection. Since the trial point `z_k = w_k + α_k d_k` is generally infeasible, the convex combination inherited that infeasibility and violated the framework invariant `x_k ∈ X`. The fix wraps the combination in `P_X`. Regression test pins the box-feasibility invariant.
- `step!` docstring was attached to the wrong function (`_fire!`) due to source-file interleaving; now correctly attached to `step!(::DFProjectionCache)`.
- Broken `@ref` link in the `AbstractDFProjection` docstring (referenced `docs/src/algorithm.md` as a path rather than a Documenter cross-reference).

### Compatibility
- Julia ≥ 1.10
- `SciMLBase` v2.53+
- `CommonSolve` v0.2.x
- `LineSearch` v0.1.x (new direct dependency)
- `Printf` (stdlib; used by `LoggingCallback`)

## [0.1.0] — 2026-05-17

First public release. A configurable derivative-free projection algorithm
for constrained nonlinear equations $F(x) = 0$ on a closed convex set $X$.

### Added
- `DFProjection` algorithm with pluggable components:
  - **Search direction**: `AbstractSearchDirection`, with `SpectralThreeTerm` as the default and the `direction!(d, rule, ctx)` user-extension API.
  - **Line search**: `AbstractDFLineSearch`, with seven variants `LSI`–`LSVII` and the `gamma_k(rule, F_z)` user-extension API.
  - **Inertial rule**: `AbstractInertialRule`, with `Inertial(θ)` and `NoInertial()`.
  - **Constraint set**: `AbstractConstraintSet`, with `RealSpace`, `BoxSet`, `HalfSpace`, `CappedBox`, `Intersection` (Dykstra), `UserSet`.
  - **Stopping criteria**: `AbstractStoppingCriterion`, with `AbsResidualTol`, `RelResidualTol`, `StepNormTol`, `DirectionNormTol`, `MaxIters`, `MaxTime`, `MaxFEvals`, `UserStop`, and the `AnyOf` composite.
- Assumption traits: `monotonicity_required`, `pseudomonotonicity_sufficient`, `convex_set_required`.
- Three documented entry points:
  - **SciML high-level**: `solve(prob, alg)` where `prob::NonlinearProblem`.
  - **SciML manual loop**: `init(prob, alg)` → `step!(cache)` → `solve!(cache)`.
  - **Standalone**: `solve_df(F, x0, alg)` returning a `DFSolution`.
- Documenter docs deployed at <https://mmogib.github.io/DFMethods.jl/>.
- Test suite: 127 tests covering all components and the full SciMLBase integration surface.

### Known limitations
- Element type is fixed to `Float64`; `Float32`, `BigFloat`, `Complex`, and static arrays are not supported in this release.
- The trial-point → projection step (Solodov–Svaiter hyperplane) is hardcoded; pluggable alternative strategies are planned for v0.2.
- No built-in history or logging callbacks; planned for v0.2.

### Compatibility
- Julia ≥ 1.10
- `SciMLBase` v2.x
- `CommonSolve` v0.2.x

[0.2.0]: https://github.com/mmogib/DFMethods.jl/releases/tag/v0.2.0
[0.1.0]: https://github.com/mmogib/DFMethods.jl/releases/tag/v0.1.0
