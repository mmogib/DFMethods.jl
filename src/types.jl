# types.jl — Abstract algorithm supertype.

"""
    AbstractDFProjection

Supertype for all derivative-free projection algorithms in DFMethods.
Concrete subtypes (e.g., `DFProjection` in `algorithm.jl`) wire together
a search direction, a line search, an inertial rule, and a constraint set.

Subtypes `SciMLBase.AbstractNonlinearAlgorithm`, so every concrete
algorithm participates in `solve(prob::NonlinearProblem, alg; kwargs...)`
out of the box.

Convergence assumptions (e.g., monotonicity of ``\\psi``, closed convex
``X``) are documented in [`docs/src/algorithm.md`](@ref). The library
does not enforce them at runtime — algorithms iterate regardless and
either converge or terminate via a stopping criterion.
"""
abstract type AbstractDFProjection <: SciMLBase.AbstractNonlinearAlgorithm end
