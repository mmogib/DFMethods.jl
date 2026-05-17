# Changelog

All notable changes to **DFMethods.jl** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Pre-v0.2 work in progress — see project notes for scope.

## [0.1.0] — 2026-05-17

First public release. Implements **UIDFPAF** (Unified Inertial Derivative-Free
Projection Algorithmic Framework) of Ibrahim, Alshahrani, Al-Homidan
(*Journal of Optimization Theory and Applications* **208**:11, 2026)
for constrained nonlinear equations F(x) = 0 on a closed convex set X.

### Added
- `DFProjection` algorithm with pluggable components:
  - **Search direction**: `AbstractSearchDirection`, with `SpectralThreeTerm` (paper default) and the `direction!(d, rule, ctx)` user-extension API.
  - **Line search**: `AbstractDFLineSearch`, with `LSI`–`LSVII` (seven variants from the paper) and the `gamma_k(rule, F_z)` user-extension API.
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

### Citation
If you use DFMethods.jl in research, please cite the source paper:
> Ibrahim, A. H., Alshahrani, M., & Al-Homidan, S. (2026).
> *A Unified Derivative-Free Projection Framework for Convex-Constrained Nonlinear Equations.*
> Journal of Optimization Theory and Applications **208**:11.
> [doi:10.1007/s10957-025-02826-x](https://doi.org/10.1007/s10957-025-02826-x)

[Unreleased]: https://github.com/mmogib/DFMethods.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mmogib/DFMethods.jl/releases/tag/v0.1.0
