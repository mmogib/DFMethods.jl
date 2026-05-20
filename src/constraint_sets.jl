# constraint_sets.jl — Constraint sets and exact projections.
#
# Used by:
#   - The algorithm: the feasible set X for the problem.
#   - The projection step: ad-hoc sets like the hyperplane H_k (= a
#     HalfSpace) and the intersection X ∩ H_k.

"""
    AbstractConstraintSet

Supertype for all constraint sets. Concrete subtypes implement
`project!(y, x, set)` that writes the orthogonal projection of `x` onto
the set into `y` and returns `y`.
"""
abstract type AbstractConstraintSet end

"""
    project!(y::AbstractVector, x::AbstractVector, set::AbstractConstraintSet) -> y

In-place orthogonal projection of `x` onto `set`. Writes the projection
into `y` and returns `y`. Each concrete [`AbstractConstraintSet`](@ref)
subtype provides its own method; see the individual set docstrings for
the projection formula or algorithm used.

For an allocating wrapper, see [`project`](@ref).
"""
function project! end

# ============================================================================
# RealSpace: no constraint (projection is identity)
# ============================================================================

"""
    RealSpace()

The whole of ``\\mathbb{R}^n`` — projection is the identity. Use to
disable constraints (e.g., for unconstrained comparison with
NonlinearSolve.jl's `SimpleDFSane`).

```jldoctest
julia> project([1.0, -2.0, 3.0], RealSpace())
3-element Vector{Float64}:
  1.0
 -2.0
  3.0
```
"""
struct RealSpace <: AbstractConstraintSet end

project!(y::AbstractVector, x::AbstractVector, ::RealSpace) = (y .= x; return y)

# ============================================================================
# BoxSet: element-wise lower / upper bounds
# ============================================================================

"""
    BoxSet(lower, upper)

Box constraint ``\\{x : \\text{lower}_i \\le x_i \\le \\text{upper}_i\\}``.
`lower` and `upper` are vectors of equal length with `lower[i] ≤ upper[i]`
element-wise. Use `BoxSet(fill(a, n), fill(b, n))` for uniform scalar bounds.

```jldoctest
julia> project([2.0, -3.0, 0.5], BoxSet([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]))
3-element Vector{Float64}:
  1.0
 -1.0
  0.5
```
"""
struct BoxSet{T<:AbstractFloat} <: AbstractConstraintSet
    lower::Vector{T}
    upper::Vector{T}
    function BoxSet(lower::Vector{T}, upper::Vector{T}) where {T<:AbstractFloat}
        length(lower) == length(upper) ||
            throw(ArgumentError("BoxSet: bounds length mismatch"))
        all(lower .<= upper) ||
            throw(ArgumentError("BoxSet: lower must be ≤ upper element-wise"))
        return new{T}(lower, upper)
    end
end

function project!(y::AbstractVector, x::AbstractVector, set::BoxSet)
    @inbounds @simd for i in eachindex(x)
        y[i] = clamp(x[i], set.lower[i], set.upper[i])
    end
    return y
end

# ============================================================================
# HalfSpace: {x : a' x ≤ c}
# ============================================================================

"""
    HalfSpace(a, c)

Half-space ``\\{x : a^\\top x \\le c\\}``. `a` is the normal vector (must
be nonzero), `c` the offset. Projection:

```
y = x                          if a' x ≤ c
y = x − ((a' x − c)/‖a‖²) a   otherwise
```

The hyperplane ``H_k`` in the Solodov–Svaiter step is a `HalfSpace`.
"""
struct HalfSpace{T<:AbstractFloat} <: AbstractConstraintSet
    a::Vector{T}
    c::T
    a_norm_sq::T   # cached ‖a‖²
    function HalfSpace(a::Vector{T}, c::T) where {T<:AbstractFloat}
        n2 = sum(abs2, a)
        n2 > 0 || throw(ArgumentError("HalfSpace: normal vector must be nonzero"))
        return new{T}(a, c, n2)
    end
end

function project!(y::AbstractVector, x::AbstractVector, set::HalfSpace)
    s = zero(eltype(y))
    @inbounds for i in eachindex(x)
        s += set.a[i] * x[i]
    end
    if s <= set.c
        @inbounds @simd for i in eachindex(x)
            y[i] = x[i]
        end
    else
        t = (s - set.c) / set.a_norm_sq
        @inbounds @simd for i in eachindex(x)
            y[i] = x[i] - t * set.a[i]
        end
    end
    return y
end

# ============================================================================
# Intersection: project onto S1 ∩ S2 via Dykstra's algorithm
# ============================================================================

"""
    Intersection(set1, set2; maxiter=200, tol=1e-12)

Intersection of two closed convex sets. Projection uses Dykstra's
algorithm (alternating projection with two correction sequences). For
``\\text{Box} \\cap \\text{HalfSpace}`` (the polyhedral set
``\\Omega`` realized by [`CappedBox`](@ref)), Dykstra converges geometrically.

For workloads that need general polyhedral projection, a direct QP
solver (JuMP + HiGHS) will be added in Phase 2 as an alternative backend.
"""
struct Intersection{S1<:AbstractConstraintSet, S2<:AbstractConstraintSet} <: AbstractConstraintSet
    set1::S1
    set2::S2
    maxiter::Int
    tol::Float64
    function Intersection(s1::AbstractConstraintSet, s2::AbstractConstraintSet;
                          maxiter::Int=200, tol::Float64=1e-12)
        return new{typeof(s1), typeof(s2)}(s1, s2, maxiter, tol)
    end
end

function project!(y::AbstractVector, x::AbstractVector, set::Intersection)
    T = eltype(y)
    n = length(x)
    p     = zeros(T, n)
    q     = zeros(T, n)
    z     = similar(x)
    y_old = similar(x)
    y .= x
    @inbounds for _ in 1:set.maxiter
        copyto!(y_old, y)
        # z = y + p ; y_new = P_{S1}(z) ; p_new = z - y_new
        @. z = y + p
        project!(y, z, set.set1)
        @. p = z - y
        # z = y + q ; y_new = P_{S2}(z) ; q_new = z - y_new
        @. z = y + q
        project!(y, z, set.set2)
        @. q = z - y
        # convergence on ‖y - y_old‖
        diff = zero(T)
        for i in 1:n
            diff += abs2(y[i] - y_old[i])
        end
        if sqrt(diff) < set.tol
            break
        end
    end
    return y
end

# ============================================================================
# CappedBox: Ω(a, b, c) = [a, b]^n ∩ {x : Σ x_i ≤ c}
# ============================================================================

"""
    CappedBox(a, b, c)

The polyhedral set ``\\Omega(a, b, c) = \\{x \\in \\mathbb{R}^n : a \\le x_i \\le b,\\ \\sum_i x_i \\le c\\}``
— a uniform-bound box intersected with a single budget-style halfspace.
`a, b, c` are scalars; the dimension `n` is inferred from the input
vector at projection time. This set is a common experimental domain in
the convex-constrained nonlinear-equations literature.

# Projection
Closed-form via bisection on a 1D Lagrange multiplier:
1. Compute the box projection ``y_B = \\mathrm{clamp}(x, a, b)``.
2. If ``\\mathbf{1}^\\top y_B \\le c`` → return ``y_B`` (constraint inactive).
3. Otherwise find ``λ^* > 0`` such that
   ``\\sum_i \\mathrm{clamp}(x_i - λ^*, a, b) = c``, then return
   ``\\mathrm{clamp}(x - λ^* \\mathbf{1}, a, b)``.

The sum is piecewise-linear monotone-decreasing in `λ`, so bisection
converges in `O(log(1/tol))` steps.

# Feasibility
The set is empty iff `n·a > c`. `project!` throws `ArgumentError` in
that case (the feasibility check requires `n`, so it cannot happen at
construction time).
"""
struct CappedBox{T<:AbstractFloat} <: AbstractConstraintSet
    a::T
    b::T
    c::T
    function CappedBox(a::T, b::T, c::T) where {T<:AbstractFloat}
        a <= b || throw(ArgumentError("CappedBox: need a ≤ b"))
        return new{T}(a, b, c)
    end
end

CappedBox(a::Real, b::Real, c::Real) = CappedBox(float(a), float(b), float(c))

function project!(y::AbstractVector, x::AbstractVector, set::CappedBox)
    n = length(x)
    a, b, c = set.a, set.b, set.c
    n * a > c && throw(ArgumentError("CappedBox: infeasible (n·a = $(n*a) > c = $c)"))

    # Step 1: try plain box projection
    s0 = zero(eltype(y))
    @inbounds for i in 1:n
        s0 += clamp(x[i], a, b)
    end
    if s0 <= c
        @inbounds @simd for i in 1:n
            y[i] = clamp(x[i], a, b)
        end
        return y
    end

    # Step 2: bisection on λ ≥ 0 to enforce Σ clamp(x_i - λ, a, b) = c.
    # Initial bracket:
    #   λ_lo = 0   (S(0) > c, by the case 1 fall-through)
    #   λ_hi: pick large enough that S(λ_hi) ≤ c. Any λ ≥ max_i(x_i - a)
    #         makes every component clamp to `a`, giving S = n·a ≤ c.
    x_max = -Inf
    @inbounds for i in 1:n
        x_max = max(x_max, x[i] - a)
    end
    λ_lo = zero(eltype(y))
    λ_hi = max(x_max, oneunit(eltype(y)))

    for _ in 1:100
        λ = (λ_lo + λ_hi) / 2
        s = zero(eltype(y))
        @inbounds for i in 1:n
            s += clamp(x[i] - λ, a, b)
        end
        if abs(s - c) < 1e-12 || (λ_hi - λ_lo) < 1e-14
            λ_lo = λ_hi = λ
            break
        elseif s > c
            λ_lo = λ
        else
            λ_hi = λ
        end
    end

    λ_final = (λ_lo + λ_hi) / 2
    @inbounds @simd for i in 1:n
        y[i] = clamp(x[i] - λ_final, a, b)
    end
    return y
end

# ============================================================================
# UserSet: arbitrary user-supplied projection
# ============================================================================

"""
    UserSet(proj!)

Constraint set defined by a user-supplied in-place projection
`proj!(y, x)` that writes the projection of `x` onto the set into `y`.
The user is responsible for ensuring `proj!` projects onto a closed
convex set — otherwise the algorithm's convergence guarantees do not hold.
"""
struct UserSet{F} <: AbstractConstraintSet
    proj!::F
end

project!(y::AbstractVector, x::AbstractVector, set::UserSet) =
    (set.proj!(y, x); return y)

# ============================================================================
# Convenience: out-of-place projection
# ============================================================================

"""
    project(x, set) -> Vector

Allocating out-of-place projection. For tests and small problems.
Inside the algorithm hot loop, always use `project!`.
"""
project(x::AbstractVector, set::AbstractConstraintSet) =
    project!(similar(x), x, set)
