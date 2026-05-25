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

Each subsection points to a runnable file in `examples/` that
demonstrates a custom subtype end-to-end. Copy one of those files as a
template for your own extension.

## Preliminaries: contract surface (`ctx`, `cache`, per-solve state)

DFMethods's seven extension points use three different conventions for
passing per-iteration state to user code. This section is the canonical
reference for all of them; later sections describe the slots themselves
and reference back here for field-list details.

### Which contract receives what

| Slot | Receives | Function signature | Pattern |
|---|---|---|---|
| §1 Line search        | (CommonSolve cache) | `init(prob, alg, fu, u)` then `solve!(cache, u, du)` | Two-method `CommonSolve` contract |
| §2 Search direction   | `ctx` (NamedTuple) | `direction!(d, rule, ctx)`              | In-place write into `d` |
| §3 Iterate update     | `ctx` (NamedTuple) | `update_iterate!(x_new, rule, ctx)`     | In-place write into `x_new` |
| §4 Inertial rule      | plain scalars       | `inertial_coef(rule, k, xk, xkm1)`      | Returns `Float64` |
| §5 Constraint set     | (nothing)           | `project!(y, x, set)`                   | In-place write into `y` |
| §6 Stopping criterion | `cache`             | `on_event!(crit, cache, event)`         | Returns `(Bool, Symbol)` |
| §7 Callback           | `cache`             | `on_event!(cb, cache, event)`           | Returns `nothing` |

ctx-based contracts (§2, §3) and cache-based contracts (§6, §7) see
different objects with different lifetimes: `ctx` is a small per-iteration
NamedTuple, while `cache` is the live mutable [`DFProjectionCache`](@ref).
Plain-argument contracts (§4, §5) get only what they need and no
ambient state. Line search (§1) is its own pattern entirely.

### `direction!` ctx — 8 fields

Constructed by the framework once per iteration, **before** the line
search. The vector fields are aliases into [`DFProjectionCache`](@ref)
buffers, which are `Vector{T}` parametric on `T = eltype(x0) <:
AbstractFloat` (with `Float64` fallback if `eltype(x0)` is not floating).
A `NamedTuple` with:

| Field | Type | Meaning |
|---|---|---|
| `Fw`              | `Vector{T}`   | ``F(w_k)`` |
| `Fw_prev`         | `Vector{T}`   | ``F(w_{k-1})`` |
| `w`               | `Vector{T}`   | inertial point ``w_k = x_k + θ_k(x_k - x_{k-1})`` |
| `w_prev`          | `Vector{T}`   | previous inertial point ``w_{k-1}`` |
| `d_prev`          | `Vector{T}`   | previous direction ``d_{k-1}`` |
| `k`               | `Int`         | iteration index (0-based) |
| `α_prev`          | `T`           | previous line-search step ``α_{k-1}``; `one(T)` at `k = 0` |
| `direction_state` | rule-specific | per-solve state from `init_state(::AbstractSearchDirection, ...)`; `nothing` if undeclared |

### `update_iterate!` ctx — 11 fields

Constructed once per iteration, **after** the line search has returned.
The vector fields alias into [`DFProjectionCache`](@ref) buffers,
parametric on `T` as documented for `direction!` above. A `NamedTuple`
with:

| Field | Type | Meaning |
|---|---|---|
| `w`             | `Vector{T}`               | inertial point |
| `d`             | `Vector{T}`               | search direction (just computed by `direction!`) |
| `α`             | `T`                       | current line-search step size |
| `z`             | `Vector{T}`               | trial point ``z_k = w + α·d`` |
| `Fw`            | `Vector{T}`               | ``F(w)`` |
| `Fz`            | `Vector{T}`               | ``F(z)`` |
| `set`           | `AbstractConstraintSet`   | resolved feasibility set |
| `k`             | `Int`                     | iteration index |
| `ζ`             | `Float64`                 | the algorithm's ``ζ`` parameter (used by Solodov–Svaiter projection) |
| `inner_maxiter` | `Int`                     | inner-projection iteration cap |
| `state`         | rule-specific             | per-solve state from `init_state(::AbstractIterateUpdate, ...)`; `nothing` if undeclared |

### Why the two ctxs differ

`direction!` fires before the line search and produces `d`. It needs the
history fields (`Fw_prev`, `w_prev`, `d_prev`, `α_prev`) used by
difference-based directions, but it cannot see line-search outputs that
have not yet been computed. `update_iterate!` fires after the line
search; it has the line-search outputs (`d`, `α`, `z`, `Fz`) and the
projection parameters (`set`, `ζ`, `inner_maxiter`) it needs, but the
history is irrelevant since it only operates on the current step.

### The `k = 0` invariant

At iteration 0, `direction!`'s ctx contains meaningful values only in
`Fw`, `w`, and `k`. The history fields (`Fw_prev`, `w_prev`, `d_prev`)
are initialized to zero vectors but rules should not depend on this. The
canonical `k = 0` fallback for a difference-based direction is
`d = -Fw` (see [`SpectralThreeTerm`](@ref) and the MPRPL example).
`α_prev` is initialized to `1.0`.

### The `cache` object (stopping criteria, callbacks)

`cache` is the live [`DFProjectionCache`](@ref) — the same mutable
struct returned by `init(prob, alg)`. Stopping criteria and callbacks
receive it (read-only by convention) at four lifecycle events:

| Event | When it fires | Cache state |
|---|---|---|
| `:initialize`       | once, at the end of `init_cache`        | ``x = x_0`` (projected onto `set`); `Fw`, `Fz`, `α_prev` initialized; `k = 0` |
| `:post_linesearch`  | each iter, after line search            | `cache.Fz` fresh; `cache.α_prev` is the just-accepted step; `cache.z` is the trial point |
| `:post_iter`        | each iter, after iterate update         | `cache.x` is the new iterate; `cache.k` has been incremented |
| `:terminate`        | once, on any termination path           | final state; `cache.retcode` set |

Publicly-readable cache fields: `x`, `x_prev`, `w`, `w_prev`, `z`, `d`,
`d_prev`, `Fw`, `Fw_prev`, `Fz`, `k`, `n_evals`, `converged`, `done`,
`retcode`, `resid`, `F0_norm`, `t_start`, `α_prev`. See the source of
[`DFProjectionCache`](@ref) for the full struct definition.

### Plain-argument contracts (inertial, constraint set)

`inertial_coef(rule, k, xk, xkm1)` and `project!(y, x, set)` receive no
ambient context — only the inputs they directly need. This keeps them
simple, but it also means they cannot react to F values or iteration
state. An inertial rule that needs history (e.g. a schedule that adapts
to stalled progress) can hold state on the rule struct itself via `Ref`
or `Vector`.

### Per-solve `state` via `init_state`

Components that need per-solve scratch (allocated once, reused across
iterations of one solve) declare:

```julia
DFMethods.init_state(rule::MyRule, prob, x0, alg) -> state
```

The default (no method defined) is `nothing`. The returned `state` is
held on `DFProjectionCache` in the slot matching the component's role:
`cache.direction_state`, `cache.iterate_update_state`, or
`cache.stopping_state`. Surfacing depends on the contract:

- **Search direction** (§2): available as `ctx.direction_state` in
  `direction!`.
- **Iterate update** (§3): available as `ctx.state` in
  `update_iterate!`. See the Mann iteration example.
- **Stopping criteria, callbacks** (§6, §7): not surfaced through
  `on_event!`. Persistent state should be held on the rule struct
  itself (mutable struct with `Ref`/`Vector` fields).

The field name differs between the two surfaced contracts for historical
reasons: `direction!`'s ctx uses the prefixed `direction_state` (added in
v0.3.0), while `update_iterate!`'s ctx keeps the unprefixed `state` from
the original v0.2.0 surface.

### Mutation policy

- `ctx` and `cache` fields are read-only inputs. Custom rules must not
  mutate them.
- Outputs are written into the explicitly-passed argument: `d` for
  `direction!`, `x_new` for `update_iterate!`, `y` for `project!`,
  `w` for `apply_inertial!`.
- New NamedTuple fields may be added to `ctx` in future releases without
  breaking custom rules (rules ignore fields they don't read).
- Inside a custom rule, `keys(ctx)` returns the live field tuple. The
  schemas above are also pinned in `test/runtests.jl` so this
  documentation and the implementation cannot drift apart silently.

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

### Custom line search

A user-defined line search subtypes `LineSearch.AbstractLineSearchAlgorithm`,
declares a mutable cache, and implements `CommonSolve.init` and
`CommonSolve.solve!`. The cache is the natural place to carry state that
persists across `solve!` calls within one solve — for example, the
residual-norm window of a non-monotone Armijo rule:

```julia
Base.@kwdef struct NonmonotoneArmijo <: LineSearch.AbstractLineSearchAlgorithm
    σ::Float64 = 1e-4   # stays Float64 (algorithm hyperparam, Approach A)
    ρ::Float64 = 0.6
    memory::Int = 5
    maxbt::Int = 50
end

# Cache parametric on element type T derived from the problem's u0.
mutable struct NonmonotoneCache{T<:AbstractFloat, F} <: LineSearch.AbstractLineSearchCache
    F::F
    alg::NonmonotoneArmijo
    z_cache::Vector{T}
    fu_cache::Vector{T}
    history::Vector{T}   # recent ‖F(z)‖² values
    n_evals::Int
end

CommonSolve.init(prob, alg::NonmonotoneArmijo, fu, u) = NonmonotoneCache(...)
CommonSolve.solve!(cache::NonmonotoneCache, u, du)    = ...  # backtrack + history update
```

The acceptance rule replaces the LHS reference value with the maximum
``\|F\|^2`` over the most recent `M` accepted residuals; monotone Armijo
is recovered by `M = 1`. See `examples/nonmonotone_armijo.jl` for the
full runnable file.

Line-search rules from the broader SciML ecosystem (e.g.
`LineSearch.LiFukushimaLineSearch`) plug in directly via the same
contract.

## 2. Search direction

A search direction subtypes [`AbstractSearchDirection`](@ref) and
implements one method:

```julia
DFMethods.direction!(d::AbstractVector, rule::MyDirection, ctx) -> d
```

### Convergence contract

Custom directions must produce $d_k$ satisfying two bounds, which DFMethods
does **not** check at runtime — they are the rule author's responsibility:

- **Sufficient descent**: $-F(w_k)^\top d_k \ge c\, \|F(w_k)\|^2$ for some
  constant $c > 0$ independent of $k$.
- **Bounded growth**: $\|d_k\| \le \bar c\, \|F(w_k)\|$ for some constant
  $\bar c > 0$ independent of $k$.

The built-in [`SpectralThreeTerm`](@ref) satisfies both: the
$[\alpha_{\min}, \alpha_{\max}]$ clamp on its spectral coefficient
$\vartheta_k^I$ gives $-F(w_k)^\top d_k = \vartheta_k^I \|F(w_k)\|^2 \ge
\alpha_{\min} \|F(w_k)\|^2$, and the $v_k$ denominator together with the
same clamp yields $\|d_k\| \le (\alpha_{\max} + 2/\bar\alpha_1)\|F(w_k)\|$.
Violating either bound forfeits the convergence guarantee (see §8).

### `direction!` `ctx`

The context `ctx` is a `NamedTuple` carrying the per-iteration inputs:

| Field                  | Description                                                  |
|------------------------|--------------------------------------------------------------|
| `ctx.Fw`               | $F$ at the inertial point $w_k$                              |
| `ctx.Fw_prev`          | $F$ at the previous inertial point $w_{k-1}$                 |
| `ctx.w`, `ctx.w_prev`  | the inertial points themselves                               |
| `ctx.d_prev`           | previous direction $d_{k-1}$                                 |
| `ctx.k`                | iteration index (0-based)                                    |
| `ctx.α_prev`           | previous line-search step $\alpha_{k-1}$ (`1.0` at `k = 0`)  |
| `ctx.direction_state`  | per-solve state from `init_state(::MyDirection, ...)`; `nothing` if undeclared |

The rule mutates `d` in place. At `k = 0`, only `ctx.Fw` is meaningful;
all other fields may be uninitialized. The framework can grow `ctx` with
new fields in later releases without breaking existing rules. See
`examples/mprpl_direction.jl` for a runnable end-to-end example using
the MPRPL direction (a difference-based conjugate-gradient-style rule).
A more elaborate example — the NHSCG direction of
[Abubakar et al. 2022](https://doi.org/10.1016/j.matcom.2021.07.005) — is
reproduced here in abbreviated form:

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

(The convergence contract for `d_k` is stated at the top of this section.
See §8 for the consequences of violating it.)

## 3. Iterate update

The post-line-search update — steps 6–7 of the algorithm — factors out as
an [`AbstractIterateUpdate`](@ref) strategy. Each subtype implements

```julia
DFMethods.update_iterate!(x_new::AbstractVector, rule::MyUpdate, ctx) -> x_new
```

The `ctx` NamedTuple here has 11 fields (`w`, `d`, `α`, `z`, `Fw`, `Fz`,
`set`, `k`, `ζ`, `inner_maxiter`, `state`); see *Preliminaries* above for
the full table and the per-iteration ordering.

Three built-in strategies ship:

| Strategy                          | Update rule                                                  |
|-----------------------------------|---------------------------------------------------------------|
| [`SolodovSvaiterProjection`](@ref) (default) | $x_{k+1} = \tilde P_{X \cap H_k}(w - \lambda_k F(z);\, \varepsilon_k)$ |
| [`DirectUpdate`](@ref)            | $x_{k+1} = P_X(z_k)$                                          |
| [`HalpernUpdate`](@ref)`(β)`      | $x_{k+1} = \beta\, x_0 + (1-\beta)\, z_k$ (scalar or schedule) |

`SolodovSvaiterProjection` implements the hyperplane projection scheme
of [Solodov & Svaiter (1999)](https://doi.org/10.1137/S0363012997317475).
`DirectUpdate` is the simplest baseline. `HalpernUpdate` admits both a
constant $\beta$ and a callable schedule $k \mapsto \beta(k)$ such as
$\beta(k) = 1/(k+2)$ for the classical
[Halpern (1967)](https://doi.org/10.1090/S0002-9904-1967-11864-0)
iteration.

### Implicit contract for `SolodovSvaiterProjection`

The hyperplane-projection update relies on the multiplier

```math
\lambda_k \;=\; \frac{F(z_k)^\top (w_k - z_k)}{\|F(z_k)\|^2}
```

being strictly positive — otherwise the target $w_k - \gamma\lambda_k F(z_k)$
lies on the wrong side of the separating hyperplane $H_k$ and the
geometry flips silently. Positivity follows whenever the line search
enforces an Armijo-style separation

```math
-F(z_k)^\top d_k \;\ge\; \sigma\, \alpha_k\, \gamma_k\, \|d_k\|^2 \;>\; 0
```

which together with $z_k = w_k + \alpha_k d_k$ gives
$F(z_k)^\top (w_k - z_k) = -\alpha_k F(z_k)^\top d_k > 0$. All three
built-in line searches ([`ConstantBacktrack`](@ref),
[`ResidualNormBacktrack`](@ref), [`AdaptiveClampedBacktrack`](@ref))
enforce this. **Custom line searches that don't satisfy the separation
will produce $\lambda_k \le 0$ silently** — there is no runtime check;
debug by logging `λ` from inside a custom `update_iterate!` or by
verifying the separation condition holds at acceptance.

### Custom iterate-update strategy

A custom rule supplies its own `update_iterate!`; if it needs per-solve
state (e.g. to track `x_k`, which `update_iterate!`'s ctx does not
expose directly), declare `init_state(::MyUpdate, prob, x0, alg)` and
access it as `ctx.state`. The Mann iteration

```math
x_{k+1} = \alpha_k\, x_k + (1 - \alpha_k)\, P_X(z_k)
```

needs `x_k` and is the canonical state-using example:

```julia
struct MannIteration{A} <: AbstractIterateUpdate
    α::A   # scalar or k -> Float64
end

# State parametric on element type T derived from the projected initial
# iterate `x` that the framework passes to init_state.
mutable struct MannState{T<:AbstractFloat}
    xk::Vector{T}
    proj_z::Vector{T}
end

function DFMethods.init_state(::MannIteration, prob, x, alg)
    T = eltype(x)
    return MannState{T}(copy(x), similar(x))
end

function DFMethods.update_iterate!(x_new, rule::MannIteration, ctx)
    T  = eltype(x_new)
    αk = T(rule.α isa Number ? rule.α : rule.α(ctx.k))
    project!(ctx.state.proj_z, ctx.z, ctx.set)
    @. x_new = αk * ctx.state.xk + (one(T) - αk) * ctx.state.proj_z
    copyto!(ctx.state.xk, x_new)
    return x_new
end
```

See `examples/mann_iteration.jl` for the full runnable file. The classical
schedule `α_k = 1/(k + 2)` ([Mann 1953](https://doi.org/10.1090/S0002-9939-1953-0054846-3))
plugs in as `MannIteration(k -> 1/(k + 2))`.

## 4. Inertial rule

The inertial step (step 1 of the algorithm) is governed by an
[`AbstractInertialRule`](@ref). The contract is the simplest in the
package — a single function returning a scalar, with no `ctx` or
`cache` arguments:

```julia
DFMethods.inertial_coef(rule::MyInertia, k::Int, xk, xkm1) -> Float64
```

The framework then computes ``w_k = x_k + θ_k\,(x_k - x_{k-1})``.

Two built-ins ship:

```@example ext
Inertial(0.25)   # θ_k = min{θ, 1/(k²‖x_k − x_{k−1}‖)}; default DFProjection setting
```

```@example ext
NoInertial()     # w_k = x_k; disables inertia
```

The empirical importance of `Inertial(θ)` is non-trivial. On Dai 2015
P4 at $n = 1000$, switching `Inertial(0.25) → NoInertial()` slowed the
MPRPL direction from a median of 76 iterations to 296 — a factor of nearly
four. `Inertial(0.25)` is a common default in the literature;
`NoInertial()` is appropriate when comparing against a paper whose
algorithm has no inertia (e.g. when reproducing
[Abubakar et al. 2022](https://doi.org/10.1016/j.matcom.2021.07.005)).

### Custom inertial rule

A custom schedule is a one-liner. For the Nesterov-style schedule
``θ_k = (k-1)/(k+α)`` used in
[Maingé (2008)](https://doi.org/10.1016/j.cam.2007.07.021):

```julia
struct NesterovInertia <: AbstractInertialRule
    α::Float64
end

DFMethods.inertial_coef(rule::NesterovInertia, k::Int, xk, xkm1) =
    k == 0 ? 0.0 : (k - 1) / (k + rule.α)
```

Plug in via `DFProjection(; inertial = NesterovInertia(3.0))`. See
`examples/mainge_inertia.jl` for the full runnable file. Stateful
schedules (e.g. an adaptive θ that reacts to stalled progress) can hold
state on the rule struct itself via `Ref`/`Vector`, since `inertial_coef`
is invoked with the rule.

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

### Custom constraint set

A real custom set has non-trivial projection algorithmics. The ℓ₁ ball
``X = \{y : \|y\|_1 \leq r\}`` uses the sorted-soft-threshold algorithm
of [Duchi et al. (2008)](https://doi.org/10.1145/1390156.1390191):

```julia
struct L1Ball <: AbstractConstraintSet
    r::Float64
end

function DFMethods.project!(y, x, set::L1Ball)
    s = sum(abs, x)
    s ≤ set.r && return (copyto!(y, x); y)
    μ = sort!(abs.(x); rev = true)
    cumsum_μ, θ = 0.0, 0.0
    for j in eachindex(μ)
        cumsum_μ += μ[j]
        cand = (cumsum_μ - set.r) / j
        μ[j] - cand > 0 && (θ = cand)
    end
    @. y = sign(x) * max(abs(x) - θ, 0.0)
    return y
end
```

See `examples/l1_ball.jl` for the full runnable file with a sanity check
on a small input and a constrained-solve demo. Use via
`ConstrainedNonlinearProblem(inner, L1Ball(r))`.

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
progress bars, NaN guards, custom logging, runtime statistics. A
[ProgressMeter.jl](https://github.com/timholy/ProgressMeter.jl)
integration sketched below is a self-contained example:

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

A representative convergence theorem for this class of algorithms
requires:

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

A representative comparison on the two
[Dai et al. 2015](https://doi.org/10.1016/j.amc.2015.08.014) problems at
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
PRP-style hybrid) exhibits identical sublinear behavior on Dai P2:
1000 iterations to reach residual $\sim 3 \times 10^{-4}$, falling short
of the paper's $10^{-5}$ tolerance. Switching to a strongly monotone problem (Abubakar Table 1 P6,
tridiagonal exponential) reduces the iteration count to 17.

**Takeaway:** validate any custom direction on a panel of problem
classes — at least one well-conditioned, one ill-conditioned, and one
with a degenerate Jacobian at the solution — before deploying it in
production.
