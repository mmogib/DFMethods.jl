```@meta
CurrentModule = DFMethods
```

# API Reference

## Index

```@index
```

## Algorithm types

```@docs
AbstractDFProjectionAlgorithm
DFProjection
DFProjectionCache
DFSolution
```

## Solving

```@docs
solve_df
solve_df!
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
AbstractDFLineSearch
LSI
LSII
LSIII
LSIV
LSV
LSVI
LSVII
gamma_k
linesearch!
```

## Inertial rules

```@docs
AbstractInertialRule
Inertial
NoInertial
inertial_coef
apply_inertial!
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
should_stop_at_w
should_stop_at_z
should_stop_at_end
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
