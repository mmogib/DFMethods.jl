# Changelog

All notable changes to **DFMethods.jl** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] — 2026-05-25

### Added

- **`SpectralThreeTerm`**: two new keyword arguments `alpha_min` and
  `alpha_max` (defaults `1e-10` and `1e30`) that clamp the spectral
  coefficient ``ϑ_k^I`` to the interval `[alpha_min, alpha_max]`.
  The clamp underwrites the strict-descent property
  ``F(w_k)' d_k ≤ -\alpha_{\min} \|F(w_k)\|^2`` and the trust-region
  bound ``\|d_k\| ≤ (α_{\max} + 2/\bar α_1)\,\|F(w_k)\|``. In the
  degenerate case `y_{k-1} = 0` (which forced `ϑ_k^I = 0` previously,
  giving zero descent), ``ϑ_k^I`` now falls back to `alpha_min`.
- **`SolodovSvaiterProjection`**: new keyword argument `γ::Float64 =
  1.0` (relaxation factor; must lie in `(0, 2)`). The projection
  target becomes `w − γ·λ_k·F(z_k)`, and the Dykstra tolerance scales
  to `ε_k = (ζ²/2) γ² λ_k² ‖F(z_k)‖²` so Dykstra stops at a constant
  fractional accuracy of the projection step across all `γ` values.
  `γ = 1` preserves the prior release behavior byte-for-byte. For
  `X = RealSpace` the halfspace projection cancels `γ ≤ 1`; the
  relaxation has effect for `γ > 1` and for non-trivial constraint
  sets.

### Changed

- **`SpectralThreeTerm.direction!`**: the assembled spectral
  coefficient is now `clamp(s'y / y'y, alpha_min, alpha_max)` rather
  than the unconstrained `s'y / y'y`. Default knobs are wide enough
  (`[1e-10, 1e30]`) that any well-behaved trajectory produces a
  bit-identical direction to the prior release; only the degenerate
  fallback (`y_{k-1} = 0`) changes output, replacing the previous
  zero-descent failure with a strict-descent step.
- **`SolodovSvaiterProjection`** is now `Base.@kwdef`'d with a single
  `γ::Float64 = 1.0` field. The zero-argument `SolodovSvaiterProjection()`
  constructor continues to work and constructs the default rule.

## [0.3.1] — 2026-05-24

Documentation enhancement release. Adds a new **Tutorial** page to the
manual (between Quickstart and Algorithm in the nav) demonstrating the
end-to-end solver workflow with emphasis on the callback architecture
(observer callbacks, stopping criteria as callbacks, writing a custom
callback from scratch). No source-code changes; no API additions or
removals; 242/242 tests pass unchanged.

### Added

- **`docs/src/tutorial.md`** — hands-on walkthrough (live-evaluated via
  Documenter `@example` blocks):
  - § 1 Setup — minimum-viable `solve(NonlinearProblem(F, x0), DFProjection())`
  - § 2 Choosing components — overrides for `linesearch`, `inertial`, etc.
  - § 3 Built-in observer callbacks — `LoggingCallback` (per-iter table) +
    `HistoryCallback` (collect data, plot inline convergence curve via
    Plots.jl)
  - § 4 Stopping criteria as callbacks — `AnyOf(AbsResidualTol, MaxIters,
    MaxTime)` composition; documents the `AbstractStoppingCriterion <:
    AbstractCallback` design
  - § 5 Writing a custom callback — `IterateSnapshotCallback` from scratch
    showing the `on_event!(cb, cache, event::Symbol)` contract and the
    `:initialize` / `:post_linesearch` / `:post_iter` / `:terminate` event
    order
  - § 6 Comparative sweep — small 3-problem × 3-line-search loop with
    results aggregated into a `DataFrame` and rendered inline

### Changed

- **`docs/Project.toml`**: added `Plots`, `DataFrames`, `LinearAlgebra` —
  needed by the new tutorial's `@example` blocks (rendered convergence
  plot in § 3, results table in § 6).
- **`docs/make.jl`**: added `"Tutorial" => "tutorial.md"` to the `pages`
  list, slotted between Quickstart and Algorithm.

### Compatibility

- Julia ≥ 1.10
- `SciMLBase` v2.x
- `CommonSolve` v0.2.x
- `LineSearch` v0.1.x

No source-level changes — drop-in upgrade from v0.3.0.

## [0.3.0] — 2026-05-22

Element-type genericity, direction-state ctx surfacing, documentation
professionalization, and the package's first Zenodo DOI. The
`DFProjectionCache` and all built-in components become parametric on
element type `T <: AbstractFloat`, lifting the v0.1.0 "Float64-only"
limitation. Algorithms now solve in `Float32`, `Float64`, or `BigFloat`
end-to-end (T flows from the problem's `eltype(u0)` with `Float64`
fallback for non-floating eltypes; algorithm-parameter struct fields stay
`Float64` and coerce at the boundary — matches the LineSearch.jl /
NonlinearSolve.jl / Optim.jl / DiffEq.jl convention). 242/242 tests pass.

### Added

- **Element-type genericity** (lifts v0.1.0 known limitation):
  - `DFProjectionCache{T, Alg, F, S, DirState, LSCache, IUpState, StopState}`
    parametric on `T <: AbstractFloat`.
  - All built-in components T-aware: search direction
    (`SpectralThreeTerm`), line searches (`ConstantBacktrack`,
    `ResidualNormBacktrack`, `AdaptiveClampedBacktrack`),
    iterate-update strategies (`SolodovSvaiterProjection`,
    `HalpernUpdate`, `DirectUpdate`), inertial rules (`Inertial`,
    `NoInertial`), constraint sets (`BoxSet`, `HalfSpace`, `CappedBox`,
    `Intersection`), stopping criteria, projection helper
    (`approx_project_X_halfspace!`).
  - `init_cache` derives `T = eltype(x0) <: AbstractFloat ? eltype(x0)
    : Float64` and allocates all per-iteration buffers as `Vector{T}`.
  - Public path: `solve(NonlinearProblem(F, Float32.(u0)), DFProjection())`
    produces a `Vector{Float32}` cache + `Vector{Float32}` `sol.u` +
    `Vector{Float32}` `sol.resid` end-to-end.
  - New test coverage: three `Float32` smoke testsets (default config,
    alternate components, box-constrained) + one `BigFloat` sanity
    testset (n=3, ≤5 s budget). 229 → 242 tests.
  - Five `examples/` files (`mprpl_direction.jl`, `nonmonotone_armijo.jl`,
    `mann_iteration.jl`, `l1_ball.jl`, `mainge_inertia.jl`) updated to
    the parametric pattern so they remain valid templates for users.
- **`direction!` ctx gains a `direction_state` field** (8 fields total).
  Routes `init_state(::AbstractSearchDirection, ...)` output through to
  custom directions. Closes the v0.2.1 known limitation. Two new
  schema-pinning tests (`direction!` ctx + `update_iterate!` ctx)
  catch doc/code drift via CI.
- **Runtime ctx introspection**: `keys(ctx)` returns the live field
  tuple inside any custom rule. Documented in `extending.md` §0
  "Mutation policy" alongside the schema-pinning convention.
- `docs/src/index.md` gains an `## Installation` section — canonical
  home for install instructions; previously only in the README.
- `CITATION.cff` (Citation File Format v1.2.0) at the package root.
  Lights up GitHub's *Cite this repository* button and feeds Zenodo
  metadata; carries author ORCID.
- `.zenodo.json` at the package root. Zenodo metadata override (creator
  with ORCID, keywords, license, upload_type).

### Changed

- `README.md` slimmed from 121 → 43 lines: no runnable code, no
  ecosystem-positioning table, no pluggable-components table. All
  duplicated content lives canonically in the docs site with link-list
  pointers in the README. Eliminates a recurring drift surface
  (README quickstart and docs quickstart had diverged).
- README build-status badge URL fixed: `?query=branch%3Amaster` →
  `?query=branch%3Amain` (mismatch with the actual default branch).
- `docs/src/extending.md` §0 ctx field tables now type fields as
  `Vector{T}` / `T` instead of `Vector{Float64}` / `Float64`; new
  explanatory note that `T = eltype(x0)` with `Float64` fallback.
- `CappedBox.project!` bisection tolerances are now T-adaptive via
  internal dispatch helpers. F64 behavior preserved exactly (still
  `1e-12` / `1e-14`); F32 / BigFloat use `sqrt(eps(T))` scaling.
- `_constraint_set` (the resolver for `prob.lb`/`prob.ub`) now derives
  T from `eltype(prob.u0)` and creates `BoxSet{T}` with `typemin(T)` /
  `typemax(T)` infinities (previously hard-coded `-Inf`/`+Inf`).

### Fixed

- **Dead-link leftovers from v0.2.1's docs-hygiene pass** (the
  `References` docs page was deleted but three pointers were missed):
  `src/inertial.jl:65`, `src/search_directions.jl:41`, and the
  `SpectralThreeTerm` source-file comment now carries the inline DOI
  https://doi.org/10.1007/s11075-023-01679-7 instead of pointing at a
  non-existent page.
- **Latent bug — `HalpernState.init_state` force-coerced `x0` to
  `Float64`** via `copy(collect(Float64, x0))`, even for Float32 /
  BigFloat problems. Now preserves `eltype(x)` (T-typed buffers).
  Bonus: Halpern's anchor is the *projected feasible* `x_0`, not the
  user's possibly-infeasible raw input (init_state's argument is now
  the projected `x` rather than raw `x0`).
- **Latent bug — `_wrap_problem_F` (in-place case) used `eltype(u0)`**
  for `out_buf` while the cache was always `Float64`, causing an in-place
  `F!(out, x, p)` to receive a `Vector{Int}` `out` when the user passed
  an integer-typed `u0`. After v0.3.0, `out_buf`'s element type matches
  the cache's `T` (with Int → Float64 fallback applied uniformly).
- `algorithm.jl` degenerate-residual guard `eps()` → `eps(typeof(...))`
  so the threshold respects the working precision (critical for
  `Float32` where `eps(Float64)` is below precision and the wrong
  threshold to use).
- `DFProjectionCache` docstring's type signature previously omitted
  the `S` (constraint set) parameter. Now correctly includes both
  `T` and `S`.

### Compatibility

- Julia ≥ 1.10 (unchanged).
- Direct deps unchanged from v0.2.x: `CommonSolve` v0.2.x,
  `LineSearch` v0.1.x, `SciMLBase` v2.53+.
- **Backward-compatible behavior for v0.2.x Float64 user code**: zero
  observable change. All 229 v0.2.1 tests pass unchanged; 13 new
  T-genericity tests added for a total of 242.
- **Type-level breaking**: code that explicitly dispatched on
  `DFProjectionCache{Alg, F, S, ...}` (the v0.2.x type signature
  without a leading `T`) needs updating to
  `DFProjectionCache{T, Alg, F, S, ...}`. Practically unaffected — the
  parametric type is internal-but-reachable; user code calls
  `init_cache(F, x0, alg)` and consumes a `cache` whose concrete type
  is inferred at call site.

### Known limitations

The two v0.2.1 / v0.1.0 limitations are both lifted in v0.3.0:
direction-state ctx surfacing (closed) and Float64-only element type
(closed — Float32, BigFloat, and any `T <: AbstractFloat` supported
end-to-end).

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

[0.3.2]: https://github.com/mmogib/DFMethods.jl/releases/tag/v0.3.2
[0.3.1]: https://github.com/mmogib/DFMethods.jl/releases/tag/v0.3.1
[0.3.0]: https://github.com/mmogib/DFMethods.jl/releases/tag/v0.3.0
[0.2.1]: https://github.com/mmogib/DFMethods.jl/releases/tag/v0.2.1
[0.2.0]: https://github.com/mmogib/DFMethods.jl/releases/tag/v0.2.0
[0.1.0]: https://github.com/mmogib/DFMethods.jl/releases/tag/v0.1.0
