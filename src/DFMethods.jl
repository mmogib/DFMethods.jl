module DFMethods

using LinearAlgebra
using SciMLBase

# ── Includes (dependency order) ──────────────────────────────────────────────
include("types.jl")                       # AbstractDFProjectionAlgorithm + traits
include("constraint_sets.jl")             # Sets + exact projections
include("inertial.jl")                    # Inertial extrapolation rules
include("search_directions.jl")           # AbstractSearchDirection + SpectralThreeTerm
include("line_searches.jl")               # AbstractDFLineSearch + LSI..LSVII
include("projection.jl")                  # Approximate projection onto X ∩ H_k
include("stopping_criteria.jl")           # AbstractStoppingCriterion + variants
include("algorithm.jl")                   # DFProjection + cache + step! + solve_df
include("nonlinearsolve_integration.jl")  # SciMLBase.__solve dispatch

# ── Exports ──────────────────────────────────────────────────────────────────
export
    # types.jl
    AbstractDFProjectionAlgorithm,

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

    # line_searches.jl
    AbstractDFLineSearch,
    LSI, LSII, LSIII, LSIV, LSV, LSVI, LSVII,
    gamma_k,

    # stopping_criteria.jl
    AbstractStoppingCriterion,
    AbsResidualTol, RelResidualTol,
    StepNormTol, DirectionNormTol,
    MaxIters, MaxTime, MaxFEvals,
    UserStop, AnyOf,
    should_stop_at_w, should_stop_at_z, should_stop_at_end,

    # algorithm.jl
    DFProjection, DFProjectionCache, DFSolution,
    init_cache, solve_df, solve_df!

end # module DFMethods
