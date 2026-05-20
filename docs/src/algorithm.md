```@meta
CurrentModule = DFMethods
```

# Algorithm

DFMethods solves systems of nonlinear equations with a closed convex
feasibility set, using a *derivative-free projection* family of methods.
Each outer iteration is composed of seven steps. Different choices for
each step produce different concrete algorithms; the package ships a
default and exposes every step as a pluggable component
(see [Extending](@ref)).

## Problem class

Find $u^* \in X$ such that

```math
F(u^*) = 0,
```

with $X \subset \mathbb{R}^n$ closed convex and $F : \mathbb{R}^n \to
\mathbb{R}^n$ continuous. The algorithm uses no derivative information
of $F$. Additional assumptions on $F$ (monotonicity-like conditions
linking $F$ and the solution set) are needed for the convergence
theorems; they are stated in the [Convergence](#Convergence) section
below.

## One outer iteration

Each call to the driver's `step!` performs the following six operations.

1. **Inertial extrapolation.**
   ```math
   w_k = x_k + \theta_k (x_k - x_{k-1}),
   ```
   with $\theta_k$ produced by the chosen [`AbstractInertialRule`](@ref).
   With [`NoInertial`](@ref), $w_k = x_k$.

2. **Function evaluation.** Compute $F(w_k)$. Stopping criteria registered
   on the `:post_linesearch` event may also evaluate convergence on
   $F(w_k)$ when their semantics call for it.

3. **Search direction.** Compute $d_k$ via the chosen
   [`AbstractSearchDirection`](@ref). For most convergence theorems in
   this family, the rule must produce a *sufficient-descent* direction
   with bounded norm:

   ```math
   -F(w_k)^\top d_k \geq c \|F(w_k)\|^2,
   \qquad
   \|d_k\| \leq \bar c\, \|F(w_k)\|,
   ```

   for fixed $c, \bar c > 0$.

4. **Line search.** Find a step size $\alpha_k > 0$ via the chosen
   line-search algorithm (any
   `LineSearch.AbstractLineSearchAlgorithm` from
   [LineSearch.jl](https://github.com/SciML/LineSearch.jl), including
   the built-ins below). The default [`ResidualNormBacktrack`](@ref)
   performs Armijo backtracking with $\gamma_k = \|F(z_k)\|$ inside the
   descent test; see [Extending](@ref) §1 for the contract.

5. **Trial point.** Form $z_k = w_k + \alpha_k d_k$ and evaluate $F(z_k)$.
   The default convergence criterion [`AbsResidualTol`](@ref) fires here
   when $\|F(z_k)\| \leq \epsilon$.

6. **Iterate update.** Produce $x_{k+1}$ via the chosen
   [`AbstractIterateUpdate`](@ref) strategy:

   - [`SolodovSvaiterProjection`](@ref) (default) constructs the
     separating hyperplane
     $H_k = \{x : F(z_k)^\top(x - z_k) \leq 0\}$ and sets
     ```math
     x_{k+1} = \tilde P_{X \cap H_k}\!\bigl(w_k - \lambda_k F(z_k);\ \epsilon_k\bigr),
     ```
     with $\lambda_k = F(z_k)^\top(w_k - z_k) / \|F(z_k)\|^2$ and
     $\epsilon_k = \tfrac{1}{2}\zeta^2 \|\lambda_k F(z_k)\|^2$.
   - [`DirectUpdate`](@ref) sets $x_{k+1} = P_X(z_k)$.
   - [`HalpernUpdate`](@ref)`(β)` sets
     $x_{k+1} = P_X\!\bigl(\beta\, x_0 + (1-\beta)\, z_k\bigr)$ (with
     $\beta$ a scalar or a schedule $k \mapsto \beta(k)$).

   See [Extending](@ref) §3 for the contract on custom strategies.

## Pluggable components

| Component             | Abstract type                       | Built-in instances                                                       |
|-----------------------|-------------------------------------|--------------------------------------------------------------------------|
| Search direction      | [`AbstractSearchDirection`](@ref)   | [`SpectralThreeTerm`](@ref)                                              |
| Line search           | `LineSearch.AbstractLineSearchAlgorithm` | [`ConstantBacktrack`](@ref), [`ResidualNormBacktrack`](@ref), [`AdaptiveClampedBacktrack`](@ref) |
| Inertial rule         | [`AbstractInertialRule`](@ref)      | [`Inertial`](@ref), [`NoInertial`](@ref)                                 |
| Iterate update        | [`AbstractIterateUpdate`](@ref)     | [`SolodovSvaiterProjection`](@ref), [`DirectUpdate`](@ref), [`HalpernUpdate`](@ref) |
| Stopping / observers  | [`AbstractCallback`](@ref)          | [`AbsResidualTol`](@ref), [`MaxIters`](@ref), [`HistoryCallback`](@ref), [`LoggingCallback`](@ref), … |

The constraint set is a property of the **problem**, not the algorithm —
see [Constraint Sets](@ref).

Each line-search rule supplies its own $\gamma_k$ inside the descent
condition of step 4:

| Line search                        | $\gamma_k$                            |
|------------------------------------|----------------------------------------|
| [`ConstantBacktrack`](@ref)        | $\gamma_k \equiv 1$                    |
| [`ResidualNormBacktrack`](@ref)    | $\gamma_k = \|F(z_k)\|$                |
| [`AdaptiveClampedBacktrack`](@ref) | clamped adaptive variant               |

## Default configuration

```@setup alg
using DFMethods
```

```@example alg
DFProjection()   # all defaults
```

equivalent to:

```@example alg
DFProjection(;
    direction      = SpectralThreeTerm(; r = 0.1, alpha_bar = 1.0),
    linesearch     = ResidualNormBacktrack(; σ = 0.01, ρ = 0.6),
    inertial       = Inertial(0.25),
    iterate_update = SolodovSvaiterProjection(),
    abstol         = 1e-6,
    maxiters       = 2000,
    ζ              = 0.5,
)
```

## Convergence

A representative convergence theorem holds under the following
assumptions:

- $F$ is continuous, with $F(x)^\top (x - u^*) \geq 0$ for some
  solution $u^*$ and all $x \in \mathbb{R}^n$ (a property automatically
  satisfied whenever $F$ is monotone or pseudo-monotone);
- $X$ is closed convex;
- the search direction satisfies the sufficient-descent and boundedness
  conditions of step 3 with fixed constants $c, \bar c > 0$;
- the line-search $\gamma_k$ is bounded below by a positive constant on
  bounded subsets of $X$.

Under these assumptions the iterates produced by `step!` converge to a
solution of $F(u) = 0$ in $X$. A representative unified convergence
result covering this family of configurations is given in
[Ibrahim, Alshahrani, Al-Homidan 2026](https://doi.org/10.1007/s10957-025-02826-x),
Theorem 3.1; component-specific convergence results (Solodov–Svaiter
projection, spectral-CG directions, Halpern anchoring, etc.) appear in
the respective originating works.

The library does not enforce these assumptions at runtime. Iteration
proceeds regardless of whether they are satisfied. Users supplying custom
components are responsible for verifying that the conditions still hold.
The [Extending](@ref) page (§8) discusses empirical consequences of
violating them — notably the sublinear behavior of PRP-style search
directions on problems with a degenerate Jacobian at the solution.
