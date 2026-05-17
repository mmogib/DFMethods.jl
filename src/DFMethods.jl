module DFMethods

using LinearAlgebra
using SciMLBase
using CommonSolve
using LineSearch

# ── Includes (dependency order) ──────────────────────────────────────────────
include("types.jl")                       # AbstractDFProjection + traits
include("constraint_sets.jl")             # Sets + exact projections
include("inertial.jl")                    # Inertial extrapolation rules
include("search_directions.jl")           # AbstractSearchDirection + SpectralThreeTerm
include("line_searches.jl")               # AbstractDFLineSearch + LSI..LSVII
include("projection.jl")                  # Approximate projection onto X ∩ H_k
include("stopping_criteria.jl")           # AbstractStoppingCriterion + variants
include("algorithm.jl")                   # DFProjection + cache + step!
include("nonlinearsolve_integration.jl")  # SciMLBase.__solve dispatch

# ── Exports ──────────────────────────────────────────────────────────────────
export
    # types.jl
    AbstractDFProjection,

    # constraint_sets.jl
    AbstractConstraintSet,
    RealSpace, BoxSet, HalfSpace, CappedBox, Intersection, UserSet,
    project!, project,

    # inertial.jl
    AbstractInertialRule,
    NoInertial, Inertial,

    # search_directions.jl
    AbstractSearchDirection,
    SpectralThreeTerm,
    direction!,

    # line_searches.jl (v0.1, will drop in 3c)
    AbstractDFLineSearch,
    LSI, LSII, LSIII, LSIV, LSV, LSVI, LSVII,
    gamma_k,

    # line_searches.jl (v0.2, Section B — LineSearch.jl-aligned)
    ConstantBacktrack, ResidualNormBacktrack, AdaptiveClampedBacktrack,

    # stopping_criteria.jl
    AbstractStoppingCriterion,
    AbsResidualTol, RelResidualTol,
    StepNormTol, DirectionNormTol,
    MaxIters, MaxTime, MaxFEvals,
    UserStop, AnyOf,
    should_stop_at_w, should_stop_at_z, should_stop_at_end,

    # algorithm.jl
    DFProjection, DFProjectionCache,
    init_cache

end # module DFMethods
