```@meta
CurrentModule = DFMethods
```

# API Reference

## Index

```@index
```

## Algorithm types

```@docs
AbstractDFProjection
DFProjection
DFProjectionCache
```

## Solving

```@docs
init_cache
step!
```

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
the standard `CommonSolve.init` / `CommonSolve.solve!` contract.

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
