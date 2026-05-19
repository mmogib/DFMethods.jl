module DFMethods

using LinearAlgebra
using SciMLBase
using CommonSolve
using LineSearch

# ── Includes (dependency order) ──────────────────────────────────────────────
include("types.jl")                       # AbstractDFProjection, ConstrainedNonlinearProblem, init_state, AbstractCallback contracts
include("constraint_sets.jl")             # Sets + exact projections
include("inertial.jl")                    # Inertial extrapolation rules
include("search_directions.jl")           # AbstractSearchDirection + SpectralThreeTerm
include("line_searches.jl")               # ConstantBacktrack, ResidualNormBacktrack, AdaptiveClampedBacktrack (LineSearch.jl-aligned)
include("projection.jl")                  # Approximate projection onto X ∩ H_k
include("iterate_updates.jl")             # AbstractIterateUpdate + Solodov–Svaiter / Direct / Halpern
include("stopping_criteria.jl")           # AbstractStoppingCriterion + 9 criteria, on_event!-dispatched
include("callbacks.jl")                   # HistoryCallback, LoggingCallback, HISTORY_FIELDS
include("algorithm.jl")                   # DFProjection + cache + step!
include("nonlinearsolve_integration.jl")  # CommonSolve.init/solve! + SciMLBase.__solve dispatch

# ── Exports ──────────────────────────────────────────────────────────────────
export
    # types.jl
    AbstractDFProjection,
    ConstrainedNonlinearProblem,

    # constraint_sets.jl
    AbstractConstraintSet,
    RealSpace, BoxSet, HalfSpace, CappedBox, Intersection, UserSet,
    project!, project,

    # inertial.jl
    AbstractInertialRule,
    NoInertial, Inertial,
    inertial_coef, apply_inertial!,

    # search_directions.jl
    AbstractSearchDirection,
    SpectralThreeTerm,
    direction!,

    # line_searches.jl (LineSearch.jl-aligned)
    ConstantBacktrack, ResidualNormBacktrack, AdaptiveClampedBacktrack,

    # iterate_updates.jl
    AbstractIterateUpdate,
    SolodovSvaiterProjection, DirectUpdate, HalpernUpdate,
    update_iterate!,

    # types.jl (callback API)
    AbstractCallback,
    on_event!,

    # stopping_criteria.jl
    AbstractStoppingCriterion,
    AbsResidualTol, RelResidualTol,
    StepNormTol, DirectionNormTol,
    MaxIters, MaxTime, MaxFEvals,
    UserStop, AnyOf,

    # callbacks.jl
    HistoryCallback, LoggingCallback, HISTORY_FIELDS,

    # algorithm.jl
    DFProjection, DFProjectionCache,
    init_cache

end # module DFMethods
