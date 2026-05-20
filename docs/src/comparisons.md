```@meta
CurrentModule = DFMethods
```

# Comparisons

DFMethods.jl occupies a narrow niche in the Julia ecosystem: derivative-free
solvers for $F(u) = 0$ with general closed-convex feasibility. Several
related packages handle adjacent — but different — problem classes. This
page sketches when each is preferred and where DFMethods is the right
choice.

## At a glance

| Package                       | Problem class                  | Derivatives needed | Constraint support |
|-------------------------------|---------------------------------|--------------------|--------------------|
| `NonlinearSolve.SimpleDFSane` | $F(u) = 0$                      | No                 | Unconstrained only |
| `NLboxsolve.jl`               | $F(u) = 0$                      | Yes (Jacobian/JFNK) | Box only           |
| `ProximalAlgorithms.jl`       | $\min f(x) + g(x)$              | Subgradient / prox  | Convex (via prox)  |
| `SPGBox.jl`                   | $\min f(x)$ s.t. box            | Yes (gradient)      | Box only           |
| **DFMethods.jl**              | $F(u) = 0,\ u \in X$ closed convex | **No**           | **General convex** |

The key axes are (a) equation-solving versus optimization, (b) requirement
for derivative information, and (c) flexibility of the feasibility set.

## `NonlinearSolve.SimpleDFSane`

[SimpleNonlinearSolve.jl](https://github.com/SciML/SimpleNonlinearSolve.jl)
ships `SimpleDFSane` — Cruz & Raydan's derivative-free spectral residual
method for unconstrained nonlinear systems. The interface is identical to
DFMethods (both consume a `SciMLBase.NonlinearProblem`), so the choice
between them is purely about constraints.

```@setup cmp
using NonlinearSolve, DFMethods
F(u, p) = u .- sin.(u)
```

```@example cmp
sol = solve(NonlinearProblem(F, ones(100)), SimpleDFSane())
sol.retcode
```

`SimpleDFSane` is faster than DFMethods on **unconstrained** problems —
it does no projection work — but it cannot enforce $u \in X$. If the
problem is unconstrained, prefer `SimpleDFSane`. If it has *any*
feasibility set (even a simple box), DFMethods is the correct package.

## `NLboxsolve.jl`

[NLboxsolve.jl](https://github.com/RJDennis/NLboxsolve.jl) implements
eight box-constrained nonlinear-equation solvers (Newton-class, Jacobian-
free Newton–Krylov, …). Its problem class overlaps with DFMethods but the
algorithms are derivative-based.

```julia
using NLboxsolve
result = nlboxsolve(F, x0, lo, hi; method = :nk)  # Newton-Krylov
```

When evaluating $\nabla F$ (or applying $J \cdot v$) is cheap, NLboxsolve
will typically converge in fewer outer iterations than DFMethods. When the
problem is **derivative-free** — black-box $F$, simulation-defined $F$, or
$F$ whose Jacobian is dense and expensive — DFMethods is preferable.

Box constraints in DFMethods are passed via `NonlinearProblem(F, u0; lb,
ub)`; constraint sets beyond box (e.g., $\{x : \sum_i x_i \leq c\}$,
ball constraints, user-supplied projections) are supported only by
DFMethods through [`ConstrainedNonlinearProblem`](@ref).

## `ProximalAlgorithms.jl`

[ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)
implements forward-backward, FISTA, Douglas–Rachford, and related
proximal-gradient methods for problems of the form $\min_x f(x) + g(x)$,
where $g$ is handled through its proximal operator (which encodes the
constraint when $g = \delta_X$, the indicator of $X$).

This is **optimization**, not equation-solving. A problem that genuinely
reads as $F(u) = 0$ is typically expressed only awkwardly as a minimization
(e.g., $\min \|F(u)\|^2$), losing structure. Conversely, if the natural
formulation is $\min f(x)$ subject to constraints, then DFMethods is the
wrong tool. ProximalAlgorithms is the natural fit, especially when the
proximal operator of $g$ admits a closed form.

## `SPGBox.jl`

[SPGBox.jl](https://github.com/m3g/SPGBox.jl) is the Spectral Projected
Gradient method for **box-constrained smooth minimization**:

```julia
using SPGBox
result = spgbox!(f, g!, x0; lower = lo, upper = hi)   # f minimization
```

Same category as ProximalAlgorithms — optimization rather than
equation-solving — and same conclusion: if the problem is genuinely a
minimization with a box constraint, SPGBox is excellent. If the natural
formulation is $F(u) = 0$, recast it through DFMethods rather than
minimizing $\|F\|^2$.

## When to use DFMethods

DFMethods.jl is the right choice when **all** of the following hold:

1. The problem is a system of equations $F(u) = 0$, not an optimization
   problem.
2. The feasibility set is closed and convex but **not** a simple box.
3. Evaluating $F$ is feasible; evaluating $\nabla F$ is either impossible
   or expensive enough to dominate iteration cost.

When (2) is relaxed to "the set is a box", `NLboxsolve.jl` becomes a
strong alternative for derivative-based settings, and DFMethods remains
preferable when (3) holds. When (2) is relaxed to "no constraint",
`SimpleDFSane` is faster.

## Decision tree

```
Is the problem F(u) = 0  (not min f(u))?
├── No  ─► ProximalAlgorithms.jl, SPGBox.jl, JuMP, Optimization.jl
└── Yes
    ├── Is there a constraint u ∈ X?
    │   ├── No  ─► NonlinearSolve.SimpleDFSane
    │   └── Yes
    │       ├── Are derivatives cheap?
    │       │   ├── Yes, X is box        ─► NLboxsolve.jl
    │       │   ├── Yes, X is more general ─► (no widely-used Julia option;
    │       │   │                              consider DFMethods nonetheless)
    │       │   └── No                   ─► DFMethods.jl  ◄── this package
    │       └── Otherwise              ─► DFMethods.jl  ◄── this package
```

The shaded leaves are the niche DFMethods.jl was designed for: convex
feasibility, no derivatives, equation form. For everything else, the
ecosystem already has well-maintained alternatives.

## Further reading

For algorithm-internal context — the seven steps of one outer iteration,
the convergence theorem, and the pluggable components — see
[Algorithm](@ref). For empirical context — convergence rates and
F-evaluation counts on a 3 000-run sweep — see [Benchmarks](@ref).
