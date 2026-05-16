```@meta
CurrentModule = DFMethods
```

# Algorithm

DFMethods implements **UIDFPAF** — the *Unified Inertial Derivative-Free Projection Algorithmic Framework* of Ibrahim, Alshahrani, Al-Homidan (JOTA 2026).

## Problem class

Find $u^* \in X$ such that $\psi(u^*) = 0$, with $X \subset \mathbb{R}^n$ closed convex and $\psi$ continuous satisfying

```math
\psi(x)^\top (x - u^*) \geq 0 \quad \forall x \in \mathbb{R}^n.
```

This holds whenever $\psi$ is monotone or pseudo-monotone — the framework's convergence theorem (Ibrahim 2026 Thm 3.1) requires no more.

## One outer iteration

Each call to `step!` (or one pass of `solve_df!`) performs:

1. **Inertial extrapolation.**
   ```math
   w_k = x_k + \theta_k (x_k - x_{k-1}),
   \quad \theta_k = \min\!\Bigl\{\theta,\ \tfrac{1}{k^2 \|x_k - x_{k-1}\|}\Bigr\}
   ```

2. **Function eval.** Compute $\psi(w_k)$. Early-stop if $\|\psi(w_k)\| \leq \epsilon$.

3. **Direction.** Compute $d_k$ via the chosen [`AbstractSearchDirection`](@ref). Must satisfy sufficient descent
   ```math
   -\psi(w_k)^\top d_k \geq c \|\psi(w_k)\|^2
   ```
   and boundedness $\|d_k\| \leq \bar c \|\psi(w_k)\|$ (Ibrahim 2026 eqs. 3–4).

4. **Backtracking line search.** Find the smallest $i \geq 0$ such that $\alpha_k = \rho^i$ satisfies
   ```math
   -\psi(w_k + \alpha_k d_k)^\top d_k \geq \sigma\, \alpha_k\, \gamma_k\, \|d_k\|^2
   ```
   with $\gamma_k$ supplied by the chosen [`AbstractDFLineSearch`](@ref).

5. **Trial point.** $z_k = w_k + \alpha_k d_k$. Early-stop if $\|\psi(z_k)\| \leq \epsilon$ and $z_k \in X$.

6. **Hyperplane.** The hyperplane $H_k = \{x : \psi(z_k)^\top(x - z_k) \leq 0\}$ separates the current iterate from the solution set.

7. **Approximate projection.**
   ```math
   x_{k+1} = \tilde P_{X \cap H_k}\!\bigl(w_k - \lambda_k \psi(z_k);\ \epsilon_k\bigr)
   ```
   with $\lambda_k = \psi(z_k)^\top(w_k - z_k) / \|\psi(z_k)\|^2$ and $\epsilon_k = \tfrac{1}{2} \zeta^2 \|\lambda_k \psi(z_k)\|^2$.

## Pluggable components

| Component | Abstract type | Built-in instances |
|---|---|---|
| Search direction | [`AbstractSearchDirection`](@ref) | [`SpectralThreeTerm`](@ref) |
| Line search | [`AbstractDFLineSearch`](@ref) | [`LSI`](@ref), [`LSII`](@ref), [`LSIII`](@ref), [`LSIV`](@ref), [`LSV`](@ref), [`LSVI`](@ref), [`LSVII`](@ref) |
| Inertial rule | [`AbstractInertialRule`](@ref) | [`Inertial`](@ref), [`NoInertial`](@ref) |
| Constraint set | [`AbstractConstraintSet`](@ref) | [`RealSpace`](@ref), [`BoxSet`](@ref), [`HalfSpace`](@ref), [`Intersection`](@ref), [`CappedBox`](@ref), [`UserSet`](@ref) |

See [Extending](@ref) for how to define your own direction or line search.

## Default configuration

```julia
DFProjection()   # equivalent to:
DFProjection(;
    direction  = SpectralThreeTerm(; r = 0.1, alpha_bar = 1.0),
    linesearch = LSII(; σ = 0.01, ρ = 0.6),
    inertial   = Inertial(0.25),
    set        = RealSpace(),
    abstol     = 1e-6,
    maxiters   = 2000,
    ζ          = 0.5,
    inner_maxiter = 500,
    maxbt      = 50,
)
```

The defaults reproduce Ibrahim 2026's experimental setup with the LSII line search (best-performing variant in Tables 1–5 of the paper).

## Convergence traits

Each algorithm declares its assumptions via three trait predicates:

```julia
monotonicity_required(::AbstractDFProjectionAlgorithm)::Bool
pseudomonotonicity_sufficient(::AbstractDFProjectionAlgorithm)::Bool
convex_set_required(::AbstractDFProjectionAlgorithm)::Bool
```

[`DFProjection`](@ref) inherits the defaults (`true` for all three) from [`AbstractDFProjectionAlgorithm`](@ref). Future algorithms (e.g., non-Lipschitz variants from Ibrahim 2023) override these to widen their assumed scope.

## Reference

> Ibrahim, A. H., Alshahrani, M., & Al-Homidan, S. (2026). *A Unified Derivative-Free Projection Framework for Convex-Constrained Nonlinear Equations.* Journal of Optimization Theory and Applications, **208**:11. <https://doi.org/10.1007/s10957-025-02826-x>
