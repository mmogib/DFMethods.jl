# types.jl — Abstract algorithm type and assumption traits.

"""
    AbstractDFProjectionAlgorithm

Supertype for all derivative-free projection algorithms in DFMethods.
Concrete subtypes (e.g., `DFProjection` in `algorithm.jl`) wire together
a search direction, a line search, an inertial rule, and a constraint set.

Subtypes `SciMLBase.AbstractNonlinearAlgorithm`, so every concrete
algorithm participates in `solve(prob::NonlinearProblem, alg; kwargs...)`
out of the box.
"""
abstract type AbstractDFProjectionAlgorithm <: SciMLBase.AbstractNonlinearAlgorithm end

# ============================================================================
# Assumption Traits
# ============================================================================
#
# Each concrete algorithm declares what it requires of ψ and X. Defaults
# match the Ibrahim 2026 framework: monotonicity (or pseudo-monotonicity)
# of ψ, closed convex X.
#
# Concrete algorithms override these to widen their assumed applicability:
#   monotonicity_required(::MyAlgo) = false      # e.g., works for non-monotone ψ

"""
    monotonicity_required(alg) -> Bool

True iff the algorithm's convergence theory requires ψ to be monotone:
``(ψ(x) - ψ(z))^\\top (x - z) \\geq 0`` for all ``x, z \\in \\mathbb{R}^n``.
"""
monotonicity_required(::AbstractDFProjectionAlgorithm) = true

"""
    pseudomonotonicity_sufficient(alg) -> Bool

True iff the algorithm converges under pseudo-monotonicity of ψ
(weaker than monotonicity). When `true`, `monotonicity_required` should
be read as "monotonicity is sufficient but not strictly necessary".
"""
pseudomonotonicity_sufficient(::AbstractDFProjectionAlgorithm) = true

"""
    convex_set_required(alg) -> Bool

True iff `X` must be a closed convex set for the algorithm's convergence
analysis to hold.
"""
convex_set_required(::AbstractDFProjectionAlgorithm) = true
