# Changelog

All notable changes to **DFMethods.jl** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] — 2026-05-20

Documentation hygiene and extension-contract normalization. Removes
harness-vocabulary references that had leaked from the project's
standalone `benchmarks/` directory into the library docs and source-file
docstrings (`s30_benchmark.jl`, `benchmarks/scripts/...`,
`experiments.db`); drops the standalone `Benchmarks` and `References`
documentation pages (component citations are now inline DOI links at
the point of use); adds a canonical *Extension contracts* reference
section to `extending.md` documenting `ctx`, `cache`, and per-solve
`state`; reorganises `examples/` into five small focused files, one per
extension contract. No source-code behavior change; all 223 tests pass.

### Added
- `docs/src/extending.md` §0 *Extension contracts: ctx, cache, and
  per-solve state* — canonical reference for the three context-passing
  conventions used by the seven extension points. Documents the
  `direction!` ctx (7 fields) and `update_iterate!` ctx (11 fields) as
  distinct NamedTuples, the `cache` lifecycle events, the `init_state`
  pattern, and the mutation policy. Resolves the field-list
  inconsistency between §2 and §3 in the v0.2.0 docs.
- `docs/src/extending.md` §1, §3, §4, §5 — each gained an inline
  custom-subtype "essence" snippet (struct + key contract method)
  cross-referencing the corresponding runnable file in `examples/`.
- Four new example files in `examples/`:
  - `nonmonotone_armijo.jl` (§1): line-search cache holding a residual-
    norm window across `solve!` calls (Grippo-Lampariello-Lucidi 1986
    lineage).
  - `mann_iteration.jl` (§3): canonical `init_state` + `ctx.state`
    demonstration (Mann 1953).
  - `mainge_inertia.jl` (§4): Nesterov-style schedule
    ``θ_k = (k-1)/(k+α)`` (Maingé 2008 lineage).
  - `l1_ball.jl` (§5): non-trivial projection via the Duchi et al. 2008
    sorted-soft-threshold algorithm.

### Renamed
- `examples/extending.jl` → `examples/mprpl_direction.jl`. The file's
  remit narrows to the §2 search-direction example (MPRPL, Dai-Chen-Wen
  2015); the `LSPower` line-search example previously bundled in is
  superseded by `examples/nonmonotone_armijo.jl`.

### Removed
- `docs/src/benchmarks.md` and its entry in `docs/make.jl`. Benchmark
  data and reproducer instructions belong with the standalone
  `benchmarks/` harness, not the library docs.
- `docs/src/references.md` and its entry in `docs/make.jl`. Component
  citations are now inline at the point of use as `[Author Year](DOI)`
  links; the `HalpernUpdate` docstring carries the Halpern 1967 citation
  directly.

### Changed
- `docs/src/algorithm.md`: dropped two `[Benchmarks](@ref)` cross-refs;
  the convergence-result reference now uses an inline DOI link.
- `docs/src/comparisons.md`: "Further reading" section removed (cross-
  linked the deleted Benchmarks page).
- `docs/src/extending.md`: §1, §3 prose rewritten to reference §0 for
  the contract surface and to point at the new example files; §2's
  cross-reference updated for the file rename; six
  `benchmarks/scripts/s05*.jl` / `s06*.jl` path references replaced
  with inline DOI citations or removed.
- `docs/src/index.md`: removed `benchmarks.md` and `references.md` from
  the `@contents` Pages list; "Theoretical lineage" pointer updated to
  send readers to component docstrings.
- `src/line_searches.jl`: stripped the "(empirical winner from the s30
  benchmark…)" parenthetical from the `ResidualNormBacktrack` docstring.
- `src/constraint_sets.jl`: "unconstrained benchmarking against" →
  "unconstrained comparison with" in the `RealSpace` docstring.
- `src/iterate_updates.jl`: added a `# Reference` block to the
  `HalpernUpdate` docstring carrying the Halpern 1967 DOI.

### Known limitations (documented in §0)
- Custom search-direction rules can declare per-solve `state` via
  `init_state`, but the state is currently *not surfaced* through
  `direction!`'s ctx. Workaround: hold scratch on the rule struct via
  `Ref`/`Vector`. Routing `direction_state` through `ctx.state` is on
  the roadmap for a future minor release.

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

[0.2.1]: https://github.com/mmogib/DFMethods.jl/releases/tag/v0.2.1
[0.2.0]: https://github.com/mmogib/DFMethods.jl/releases/tag/v0.2.0
[0.1.0]: https://github.com/mmogib/DFMethods.jl/releases/tag/v0.1.0
