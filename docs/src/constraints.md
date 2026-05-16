```@meta
CurrentModule = DFMethods
```

# Constraint Sets

The feasible set $X$ is passed to [`DFProjection`](@ref) as a struct subtyping [`AbstractConstraintSet`](@ref). Every subtype implements

```julia
project!(y, x, set) -> y
```

(mutates `y` in place with the orthogonal projection of `x` onto the set; returns `y`).

## Built-in sets

| Type | Set | Projection |
|---|---|---|
| `RealSpace()` | $\mathbb{R}^n$ | Identity |
| `BoxSet(lower, upper)` | $\{x : l_i \leq x_i \leq u_i\}$ | Element-wise `clamp` |
| `HalfSpace(a, c)` | $\{x : a^\top x \leq c\}$ | Closed-form Euclidean reflection |
| `Intersection(s1, s2)` | $S_1 \cap S_2$ | Dykstra's alternating projection |
| `CappedBox(a, b, c)` | $[a,b]^n \cap \{x : \sum x_i \leq c\}$ | Bisection on a 1D Lagrange multiplier |
| `UserSet(proj!)` | Arbitrary closed convex set | Caller-supplied in-place `proj!(y, x)` |

## Examples

```julia
using DFMethods, LinearAlgebra

# Box [-1, 1]^n
B = BoxSet(fill(-1.0, 100), fill(1.0, 100))

# Halfspace {x : sum(x) ≤ 0.5}
H = HalfSpace(ones(100), 0.5)

# Their intersection — closed-form (matches Ibrahim 2026 Ω):
Ω = CappedBox(-1.0, 1.0, 0.5)

# … or generic Dykstra fallback for arbitrary convex pairs:
Ω_generic = Intersection(B, H; maxiter = 500, tol = 1e-12)

# Custom set — supply your own in-place projection (e.g. ℓ² ball of radius r)
function proj_ball!(y, x)
    n = norm(x)
    y .= n <= 1 ? x : x ./ n
    return y
end
S = UserSet(proj_ball!)
```

## Out-of-place wrapper

For tests and small problems:

```julia
project(x, set::AbstractConstraintSet) -> Vector
```

is an allocating wrapper around `project!`.

## Choosing between `Intersection` and `CappedBox`

For Ibrahim 2026's specific $\Omega(a, b, c) = [a,b]^n \cap \{\sum x_i \leq c\}$, **always prefer `CappedBox`**: it's a single closed-form bisection per projection (O(n log iter)), much faster than Dykstra's iterative alternation.

Use `Intersection` only when:

- the constraint isn't this special form, or
- you've defined a non-built-in set type as one of the two operands.

For three-way intersections, nest: `Intersection(Intersection(S1, S2), S3)`.

## Feasibility of `x0`

`init_cache` projects the initial point onto `alg.set` before the first iteration (Ibrahim 2026 assumes $x_0 \in X$). Infeasible $x_0$ is silently corrected. If you want to detect infeasible inputs, project explicitly and compare:

```julia
x0_feas = project(x0, set)
isapprox(x0, x0_feas; atol = 1e-10) || @warn "x0 was infeasible; projected onto the set"
```

## API

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
```
