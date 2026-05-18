```@meta
CurrentModule = DFMethods
```

# Extending

[`DFProjection`](@ref) is a composition of pluggable components: a search
direction, a line search, an inertial rule, an iterate-update strategy, a
set of stopping criteria, and a list of observer callbacks. The constraint
set is a property of the **problem**, not of the algorithm (see
[Constraint Sets](@ref)). Each component has a small abstract-type contract;
a user-defined type that obeys the contract drops in alongside the
built-ins. This page presents the contracts in turn, followed by a
discussion of the convergence implications of replacing a component.

## 1. Line search

The framework adopts `LineSearch.AbstractLineSearchAlgorithm` from
[LineSearch.jl](https://github.com/SciML/LineSearch.jl). A line-search
rule and its cache obey the `CommonSolve` contract:

```julia
CommonSolve.init(prob, alg::MyLineSearch, fu, u) -> cache
CommonSolve.solve!(cache, u, du)                 -> LineSearch.LineSearchSolution
```

The cache stores algorithm parameters, scratch buffers for the trial point
$z = u + \alpha\, du$ and its residual $F(z)$, and an `n_evals` counter the
outer loop reads. `solve!` returns a `LineSearch.LineSearchSolution` whose
`step_size` field is the accepted $\alpha$.

The built-ins differ only in the choice of $\gamma_k$ inside the
Armijo-style descent test
$-F(z)^\top du \geq \sigma\,\alpha\,\gamma_k\,\|du\|^2$:

```@setup ext
using NonlinearSolve, DFMethods
```

```@example ext
ConstantBacktrack(; σ=0.01, ρ=0.6)               # γ_k ≡ 1
```

```@example ext
ResidualNormBacktrack(; σ=0.01, ρ=0.6)           # γ_k = ‖F(z)‖
```

```@example ext
AdaptiveClampedBacktrack(; σ=0.01, ρ=0.6, lo=1e-4, Δ_init=10.0)
```

The package's source file `src/line_searches.jl` is the canonical template
for a custom line search; the shared `_backtrack!` driver shows the
expected backtracking and counter-update pattern. Line-search rules from
the broader SciML ecosystem (e.g. `LineSearch.LiFukushimaLineSearch`) plug
in directly via the same contract.

## 2. Search direction

A search direction subtypes [`AbstractSearchDirection`](@ref) and
implements one method:

```julia
DFMethods.direction!(d::AbstractVector, rule::MyDirection, ctx) -> d
```

The context `ctx` is a `NamedTuple` carrying the per-iteration inputs:

| Field            | Description                                                  |
|------------------|--------------------------------------------------------------|
| `ctx.Fw`         | $F$ at the inertial point $w_k$                              |
| `ctx.Fw_prev`    | $F$ at the previous inertial point $w_{k-1}$                 |
| `ctx.w`, `ctx.w_prev` | the inertial points themselves                          |
| `ctx.d_prev`     | previous direction $d_{k-1}$                                 |
| `ctx.k`          | iteration index (0-based)                                    |
| `ctx.α_prev`     | previous line-search step $\alpha_{k-1}$ (`1.0` at `k = 0`)  |

The rule mutates `d` in place. At `k = 0`, only `ctx.Fw` is meaningful;
all other fields may be uninitialized. The framework can grow `ctx` with
new fields in later releases without breaking existing rules. Examples
exercising this contract live in `examples/extending.jl` (MPRPL) and
`benchmarks/scripts/s06_abubakar2022_nhscg.jl` (Abubakar et al. 2022,
NHSCG). The latter is reproduced here in abbreviated form:

```julia
struct AbubakarNHSCG <: AbstractSearchDirection
    c1::Float64; c2::Float64; c3::Float64; c4::Float64
end

function DFMethods.direction!(d, rule::AbubakarNHSCG, ctx)
    Fw, Fw_prev, d_prev, k = ctx.Fw, ctx.Fw_prev, ctx.d_prev, ctx.k
    if k == 0
        @. d = -Fw
        return d
    end
    # ... compute β^NH, λ^NH from rule.c1..c4 (paper §2 eqs. 4–7) ...
    @. d = -(1 + λ) * Fw + β * d_prev
    return d
end
```

Convergence requires the rule to produce a direction with sufficient
descent ($-F(w)^\top d \geq c\|F(w)\|^2$) and bounded norm
($\|d\| \leq \bar c\|F(w)\|$). See §8 for the consequences of relaxing
either property.

## 3. Iterate update

The post-line-search update — steps 6–7 of the algorithm — factors out as
an [`AbstractIterateUpdate`](@ref) strategy. Each subtype implements

```julia
DFMethods.update_iterate!(x_new::AbstractVector, rule::MyUpdate, ctx) -> x_new
```

with `ctx` carrying `w`, `d`, `z`, `Fz`, `set`, `ζ`, `inner_maxiter`, `k`,
and an optional `state` slot returned by
`init_state(rule, prob, x0, alg)`.

Three built-in strategies ship:

| Strategy                          | Update rule                                                  |
|-----------------------------------|---------------------------------------------------------------|
| [`SolodovSvaiterProjection`](@ref) (default) | $x_{k+1} = \tilde P_{X \cap H_k}(w - \lambda_k F(z);\, \varepsilon_k)$ |
| [`DirectUpdate`](@ref)            | $x_{k+1} = P_X(z_k)$                                          |
| [`HalpernUpdate`](@ref)`(β)`      | $x_{k+1} = \beta\, x_0 + (1-\beta)\, z_k$ (scalar or schedule) |

`SolodovSvaiterProjection` reproduces Ibrahim 2026 step by step.
`DirectUpdate` is the simplest baseline. `HalpernUpdate` admits both a
constant $\beta$ and a callable schedule $k \mapsto \beta(k)$ such as
$\beta(k) = 1/(k+2)$ for the classical Halpern iteration.

A custom strategy supplies its own `update_iterate!`; if it needs
per-solve state, also overload `init_state(::MyUpdate, prob, x0, alg)`.

## 4. Inertial rule

The inertial step (step 1 of the algorithm) is governed by an
[`AbstractInertialRule`](@ref):

```@example ext
Inertial(0.25)   # θ_k = min{θ, 1/(k²‖x_k − x_{k−1}‖)}; default DFProjection setting
```

```@example ext
NoInertial()     # w_k = x_k; disables inertia
```

The empirical importance of `Inertial(θ)` is non-trivial. On Dai 2015
P4 at $n = 1000$, switching `Inertial(0.25) → NoInertial()` slowed the
MPRPL direction from a median of 76 iterations to 296 — a factor of nearly
four. The default setting reproduces Ibrahim 2026's experimental setup;
`NoInertial()` is appropriate when comparing against a paper whose
algorithm has no inertia (e.g. the Abubakar 2022 NHSCG demo in
`benchmarks/scripts/s06_abubakar2022_nhscg.jl`).

## 5. Constraint set

A new closed convex feasible set subtypes
[`AbstractConstraintSet`](@ref) and implements

```julia
DFMethods.project!(y::AbstractVector, x::AbstractVector, set::MySet) -> y
```

(an in-place orthogonal projection of `x` onto the set). For one-off use,
[`UserSet`](@ref) wraps a function `proj!(y, x)` without requiring a new
type. The set is supplied via the **problem**: pass `lb`/`ub` to
`NonlinearProblem` for box constraints, or wrap with
[`ConstrainedNonlinearProblem`](@ref) for anything else. See
[Constraint Sets](@ref) for the full discussion.

## 6. Stopping criteria

A stopping criterion subtypes [`AbstractStoppingCriterion`](@ref)
(which subtypes [`AbstractCallback`](@ref)) and implements

```julia
DFMethods.on_event!(crit::MyCriterion, cache, event::Symbol) -> (Bool, Symbol)
```

returning `(stopped, retcode)`. The retcode is a `SciMLBase.ReturnCode`
symbol such as `:Success`, `:MaxIters`, or `:Stalled`. Criteria fire on
one or more of the four lifecycle events `:initialize`,
`:post_linesearch`, `:post_iter`, `:terminate`; the default for any other
event is `(false, :Default)`. The built-in
[`AbsResidualTol`](@ref)`(1e-5)` fires on `:post_linesearch` when
$\|F(z_k)\| \leq 10^{-5}$.

Multiple criteria compose via [`AnyOf`](@ref): the default stopping rule
of `DFProjection(; abstol=ε, maxiters=N)` is
`AnyOf(AbsResidualTol(ε), MaxIters(N))`. Custom rules drop in identically.
The following criterion replicates the paper-exact stopping rule for the
NHSCG demo (terminate when either $F(w_k)$ or $F(z_k)$ falls below
tolerance):

```julia
struct PaperNHSCGStop <: AbstractStoppingCriterion
    tol::Float64
end

function DFMethods.on_event!(c::PaperNHSCGStop, cache, event::Symbol)
    event === :post_linesearch || return (false, :Default)
    Fw_norm = sqrt(sum(abs2, cache.Fw))
    Fz_norm = sqrt(sum(abs2, cache.Fz))
    (Fw_norm ≤ c.tol || Fz_norm ≤ c.tol) ? (true, :Success) : (false, :Default)
end

alg = DFProjection(; stopping = AnyOf(PaperNHSCGStop(1e-5), MaxIters(1000)))
```

## 7. Callbacks

Observer callbacks subtype [`AbstractCallback`](@ref) directly and
implement `on_event!` returning `nothing` (the default). Two built-in
observers ship: [`HistoryCallback`](@ref) accumulates a vector of
per-iteration `NamedTuple` rows; [`LoggingCallback`](@ref) prints a table
with header, rows (every $N$ iterations), and footer. The set of available
fields is given by [`HISTORY_FIELDS`](@ref).

```@example ext
hist   = HistoryCallback(; fields = (:k, :F_norm, :α, :n_evals))
logger = LoggingCallback(; every = 50, header_every = 0, footer = true)
alg    = DFProjection(; callbacks = AbstractCallback[hist, logger])
```

A user-defined callback wraps any side effect the four events allow —
progress bars, NaN guards, custom logging, runtime statistics. The
[ProgressMeter.jl](https://github.com/timholy/ProgressMeter.jl)
integration in `benchmarks/scripts/s06_abubakar2022_nhscg.jl` is a
self-contained example:

```julia
mutable struct ProgressMeterCallback <: AbstractCallback
    maxiters::Int
    prog::Union{Nothing, ProgressMeter.Progress}
end

function DFMethods.on_event!(cb::ProgressMeterCallback, cache, event::Symbol)
    if event === :initialize
        cb.prog = ProgressMeter.Progress(cb.maxiters; dt=0.0, output=stdout)
    elseif event === :post_iter
        ProgressMeter.next!(cb.prog)
    elseif event === :terminate
        ProgressMeter.finish!(cb.prog)
    end
    return nothing
end
```

The library has no dependency on ProgressMeter.jl; the callback lives in
the user's code.

## 8. Convergence caveats

### Violating the theoretical premises

The convergence theorem of Ibrahim 2026 requires:

- a search direction with sufficient descent
  ($-F(w)^\top d \geq c\|F(w)\|^2$) and bounded norm
  ($\|d\| \leq \bar c\|F(w)\|$);
- a line-search $\gamma_k$ bounded below by a positive constant on bounded
  subsets of $X$.

If a custom component violates either condition, the algorithm continues
to iterate but the convergence guarantee is forfeited. [`SpectralThreeTerm`](@ref)
and MPRPL both satisfy the direction conditions (proved in their
respective source papers). The built-in line searches all satisfy the
$\gamma_k$ condition on any region where $F$ is bounded away from zero.

### Convergence ≠ convergence rate

Satisfying the theorem's premises buys *eventual* convergence — not a
*rate*. A theoretically valid direction may still be orders of magnitude
slower than another on a given problem.

A representative comparison from the benchmark harness
(`benchmarks/scripts/s05_extending.jl`) on the two Dai 2015 problems at
$n = 1000$, tolerance $10^{-6}$, maxiters $5000$:

| Problem | Jacobian at solution | `SpectralThreeTerm` iters | MPRPL iters |
|---|---|---|---|
| **Dai P2:** $F_i(x) = x_i - \sin(x_i)$ | $\mathrm{diag}(1 - \cos x_i) \to 0$ as $x \to 0$ | 9–22 | **stalls at 5000, residual ~5e-5** |
| **Dai P4:** $F(x) = Ax + (e^x - 1)$ | $A + I$ (well-conditioned) | 29–136 | 60–235 ✓ |

On the well-conditioned Dai P4, MPRPL converges — 2–3× slower than
`SpectralThreeTerm` but in the same iteration scale. On Dai P2 the
Jacobian *vanishes at the solution*, and MPRPL becomes effectively
sublinear ($\|F_k\| \sim 1/\sqrt{k}$). `SpectralThreeTerm` survives because
its spectral coefficient $\vartheta_k^I = s^\top y / y^\top y$ adapts to
local curvature; MPRPL has no such adaptation and reduces to a fixed-point
iteration on this problem class.

This is consistent with the literature. Dai 2015's reported "MPRPL takes
~12 iterations on Dai P2 at $n = 1000$" uses an `atol + rtol` formula that
terminates at residual $\sim 3 \times 10^{-4}$, well above MPRPL's
accuracy ceiling on a degenerate-Jacobian problem. A tolerance of
$10^{-6}$ is two orders of magnitude tighter than the regime in which MPRPL
was originally validated.

The same NHSCG direction
([Abubakar 2022](https://doi.org/10.1016/j.matcom.2021.07.005), a
PRP-style hybrid) exhibits identical sublinear behavior on Dai P2 in
`benchmarks/scripts/s06_abubakar2022_nhscg.jl`: 1000 iterations to reach
residual $\sim 3 \times 10^{-4}$, falling short of the paper's $10^{-5}$
tolerance. Switching to a strongly monotone problem (Abubakar Table 1 P6,
tridiagonal exponential) reduces the iteration count to 17.

**Takeaway:** validate any custom direction on a panel of problem
classes — at least one well-conditioned, one ill-conditioned, and one
with a degenerate Jacobian at the solution — before deploying it in
production. `benchmarks/scripts/s05_extending.jl` is the runnable
scaffold for that comparison.
