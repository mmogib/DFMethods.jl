```@meta
CurrentModule = DFMethods
```

# API Reference

## Index

```@index
```

## Problem types

```@docs
ConstrainedNonlinearProblem
```

A standard `SciMLBase.NonlinearProblem` covers unconstrained problems and
box-constrained problems (via `prob.lb` / `prob.ub`).
[`ConstrainedNonlinearProblem`](@ref) wraps a `NonlinearProblem` together
with an [`AbstractConstraintSet`](@ref) for arbitrary closed convex feasible
sets.

## Algorithm types

```@docs
AbstractDFProjection
DFProjection
DFProjectionCache
```

## Solving

```@docs
init_cache
```

The internal `DFMethods.step!(::DFProjectionCache)` advances one outer
iteration on the inner cache and is used by `CommonSolve.step!` below.
User-level driving goes through `solve(prob, alg)` or the `init` / `step!`
/ `solve!` triplet documented under [SciML integration](#SciML-integration).

## SciML integration

```@docs
DFMethods.DFSciMLCache
CommonSolve.init(::SciMLBase.NonlinearProblem, ::DFProjection)
CommonSolve.step!(::DFMethods.DFSciMLCache)
CommonSolve.solve!(::DFMethods.DFSciMLCache)
```

## Search directions

```@docs
AbstractSearchDirection
SpectralThreeTerm
direction!
```

## Line searches

```@docs
ConstantBacktrack
ResidualNormBacktrack
AdaptiveClampedBacktrack
```

Each is a subtype of `LineSearch.AbstractLineSearchAlgorithm` (from
[`LineSearch.jl`](https://github.com/SciML/LineSearch.jl)) and implements
the standard `CommonSolve.init` / `CommonSolve.solve!` contract. User-defined
line searches obey the same contract.

## Iterate-update strategies

```@docs
AbstractIterateUpdate
SolodovSvaiterProjection
DirectUpdate
HalpernUpdate
update_iterate!
DFMethods.init_state
```

## Inertial rules

```@docs
AbstractInertialRule
Inertial
NoInertial
inertial_coef
apply_inertial!
```

## Callbacks

```@docs
AbstractCallback
on_event!
HistoryCallback
LoggingCallback
HISTORY_FIELDS
```

## Stopping criteria

```@docs
AbstractStoppingCriterion
AbsResidualTol
RelResidualTol
StepNormTol
DirectionNormTol
MaxIters
MaxTime
MaxFEvals
UserStop
AnyOf
```

## Constraint sets and projections

```@docs
AbstractConstraintSet
RealSpace
BoxSet
HalfSpace
Intersection
CappedBox
UserSet
project!
project
approx_project_X_halfspace!
```
