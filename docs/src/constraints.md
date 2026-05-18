```@meta
CurrentModule = DFMethods
```

# Constraint Sets

In DFMethods, the feasible set $X$ is a property of the **problem**, not of the
algorithm. A single configured [`DFProjection`](@ref) can therefore solve
unconstrained, box-constrained, and arbitrarily convex-constrained problems
without modification — the algorithm reads $X$ from the problem it is handed.

Three problem-construction patterns cover every case.

```@setup cons
using NonlinearSolve, DFMethods, LinearAlgebra
F(u, p) = u .- sin.(u)
```

## Unconstrained problems

A plain `SciMLBase.NonlinearProblem` has no constraints. DFMethods treats it
as $X = \mathbb{R}^n$ (the [`RealSpace`](@ref) set).

```@example cons
prob = NonlinearProblem(F, ones(100))
sol  = solve(prob, DFProjection())
sol.retcode
```

## Box-constrained problems

Box bounds are passed to `NonlinearProblem` via the standard SciML keyword
arguments `lb` and `ub`:

```@example cons
prob_box = NonlinearProblem(F, ones(100); lb = fill(-2.0, 100), ub = fill(2.0, 100))
sol_box  = solve(prob_box, DFProjection())
sol_box.retcode
```

DFMethods resolves this internally as a [`BoxSet`](@ref) and projects onto
$[\mathtt{lb}, \mathtt{ub}]$ at every iteration. No algorithm reconfiguration
is required.

## Arbitrary closed convex sets

For sets that are neither $\mathbb{R}^n$ nor a box, wrap the inner problem in
a [`ConstrainedNonlinearProblem`](@ref):

```@example cons
inner    = NonlinearProblem(F, ones(100))
set      = CappedBox(-1.0, 1.0, 0.5)        # [a,b]^n ∩ {x : ∑x_i ≤ c}
prob_cnp = ConstrainedNonlinearProblem(inner, set)
sol_cnp  = solve(prob_cnp, DFProjection())
sol_cnp.retcode
```

The all-in-one constructor

```julia
ConstrainedNonlinearProblem(F, u0; set, p = SciMLBase.NullParameters(), kwargs...)
```

builds the inner `NonlinearProblem` and the wrapper in a single call.

## Built-in sets

Every subtype of [`AbstractConstraintSet`](@ref) implements

```julia
project!(y, x, set) -> y
```

which mutates `y` in place with the orthogonal projection of `x` onto the set.

| Type                        | Set                                          | Projection                            |
|-----------------------------|----------------------------------------------|---------------------------------------|
| `RealSpace()`               | $\mathbb{R}^n$                               | Identity                              |
| `BoxSet(lower, upper)`      | $\{x : l_i \leq x_i \leq u_i\}$              | Element-wise `clamp`                  |
| `HalfSpace(a, c)`           | $\{x : a^\top x \leq c\}$                    | Closed-form Euclidean reflection      |
| `CappedBox(a, b, c)`        | $[a,b]^n \cap \{x : \sum_i x_i \leq c\}$     | Bisection on a 1D Lagrange multiplier |
| `Intersection(s1, s2)`      | $S_1 \cap S_2$                               | Dykstra's alternating projection      |
| `UserSet(proj!)`            | Arbitrary closed convex set                  | Caller-supplied in-place `proj!(y, x)` |

The first four admit closed-form projections; `Intersection` falls back to
Dykstra's iterative scheme; `UserSet` accepts any in-place projector you
supply.

## Examples

```@example cons
# Box [-1, 1]^100
B = BoxSet(fill(-1.0, 100), fill(1.0, 100))

# Halfspace {x : ∑x_i ≤ 0.5}
H = HalfSpace(ones(100), 0.5)

# Their intersection — closed-form (matches Ibrahim 2026's Ω):
Ω = CappedBox(-1.0, 1.0, 0.5)

# … or generic Dykstra fallback for arbitrary convex pairs:
Ω_generic = Intersection(B, H; maxiter = 500, tol = 1e-12)

# Custom set — supply your own in-place projection (e.g. ℓ² ball of radius 1)
function proj_ball!(y, x)
    n = norm(x)
    y .= n <= 1 ? x : x ./ n
    return y
end
S = UserSet(proj_ball!)

prob_ball = ConstrainedNonlinearProblem(F, ones(100); set = S)
sol_ball  = solve(prob_ball, DFProjection())
sol_ball.retcode
```

## Out-of-place wrapper

For tests and small problems,

```julia
project(x, set::AbstractConstraintSet) -> Vector
```

is an allocating wrapper around [`project!`](@ref).

## Choosing between `Intersection` and `CappedBox`

For Ibrahim 2026's polyhedral $\Omega(a, b, c) = [a,b]^n \cap \{\sum x_i \leq
c\}$, always prefer [`CappedBox`](@ref): one closed-form bisection per
projection ($O(n \log \text{iter})$), significantly faster than Dykstra's
alternation. Use [`Intersection`](@ref) when the constraint is not of this
special form or when at least one operand is a non-built-in set.

For three-way intersections, nest:
`Intersection(Intersection(S1, S2), S3)`.

## Feasibility of the initial point

`init_cache` projects the supplied $x_0$ onto the resolved set before the
first iteration, so the invariant $x_0 \in X$ of the convergence theorem
always holds. Infeasible inputs are silently corrected. To detect them
yourself, project explicitly and compare:

```julia
x0_feas = project(x0, set)
isapprox(x0, x0_feas; atol = 1e-10) || @warn "x0 was infeasible; projected onto the set"
```

## One algorithm, many problems

A single configured algorithm reuses across problems of every constraint
flavor:

```@example cons
alg = DFProjection()
n   = 100
x0  = ones(n)

sol1 = solve(NonlinearProblem(F, x0), alg)                                     # unconstrained
sol2 = solve(NonlinearProblem(F, x0; lb = fill(-2.0, n), ub = fill(2.0, n)), alg)  # box
sol3 = solve(ConstrainedNonlinearProblem(F, x0; set = HalfSpace(ones(n), 50.0)), alg)  # halfspace

(sol1.retcode, sol2.retcode, sol3.retcode)
```

## API

[`AbstractConstraintSet`](@ref), [`RealSpace`](@ref), [`BoxSet`](@ref),
[`HalfSpace`](@ref), [`CappedBox`](@ref), [`Intersection`](@ref),
[`UserSet`](@ref), [`project!`](@ref), [`project`](@ref),
[`ConstrainedNonlinearProblem`](@ref).
