```@meta
CurrentModule = DFMethods
```

# Extending

DFMethods has two extension points: **search directions** and **line searches**. To add either, subtype the appropriate abstract type and implement one method. The rest of the algorithm — inertia, projection, hyperplane construction, NonlinearSolve.jl integration — is inherited unchanged.

## Adding a custom search direction

**Contract:** subtype [`AbstractSearchDirection`](@ref) and implement

```julia
DFMethods.direction!(d, rule, ctx)
```

where `ctx` is a NamedTuple with these fields (see [`AbstractSearchDirection`](@ref) for the full table):

| Field | Description |
|---|---|
| `ctx.Fw` | ``\\psi`` at the inertial point ``w_k`` |
| `ctx.Fw_prev` | ``\\psi`` at the previous inertial point ``w_{k-1}`` |
| `ctx.w`, `ctx.w_prev` | the inertial points themselves |
| `ctx.d_prev` | previous direction ``d_{k-1}`` |
| `ctx.k` | iteration index (0-based) |
| `ctx.α_prev` | previous line-search step ``\\alpha_{k-1}``; `1.0` at `k = 0` |

A rule ignores whichever fields it doesn't need. The framework can add fields in future versions without breaking existing rules.

- Mutate `d` in place. `ctx` is read-only.
- For `ctx.k == 0`, only `ctx.Fw` is meaningful; the other fields may be uninitialized.
- The framework's convergence theorem (Ibrahim 2026 Thm 3.1) requires the rule to produce a **sufficient-descent** direction (`-Fw' d ≥ c‖Fw‖²` for some `c > 0`) and a **bounded** one (`‖d‖ ≤ c̄‖Fw‖`) — paper eqs. (3)–(4). The framework cannot check these for you.

### Example: MPRPL direction (Dai, Chen, Wen — AMC 2015, eq. 4.1)

```julia
using DFMethods, LinearAlgebra

struct MPRPL_Direction <: AbstractSearchDirection end

function DFMethods.direction!(d, ::MPRPL_Direction, ctx)
    Fw, Fw_prev, d_prev, k = ctx.Fw, ctx.Fw_prev, ctx.d_prev, ctx.k

    if k == 0
        @. d = -Fw
        return d
    end

    Fw_y = Fwm_sq = Fw_d = Fw_sq = 0.0
    @inbounds for i in eachindex(Fw)
        yi      = Fw[i] - Fw_prev[i]
        Fw_y   += Fw[i] * yi
        Fwm_sq += Fw_prev[i]^2
        Fw_d   += Fw[i] * d_prev[i]
        Fw_sq  += Fw[i]^2
    end

    β_PRP   = Fw_y / max(Fwm_sq, eps())
    coeff_F = -(1.0 + β_PRP * Fw_d / max(Fw_sq, eps()))

    @inbounds @simd for i in eachindex(Fw)
        d[i] = coeff_F * Fw[i] + β_PRP * d_prev[i]
    end
    return d
end
```

If your direction needs the previous step displacement (e.g. for a Dai–Liao update with `s_{k-1} = α_{k-1} · d_{k-1}`), pull it from `ctx.α_prev`:

```julia
function DFMethods.direction!(d, ::MyDirection, ctx)
    Fw, Fw_prev, d_prev, k, α_prev =
        ctx.Fw, ctx.Fw_prev, ctx.d_prev, ctx.k, ctx.α_prev
    if k == 0
        @. d = -Fw
        return d
    end
    # ... use s_{k-1} = α_prev .* d_prev wherever needed ...
end
```

## Adding a custom line search

**Contract:** subtype [`AbstractDFLineSearch`](@ref) with fields `σ::Float64` (Armijo coefficient) and `ρ::Float64` (backtracking factor), and implement

```julia
DFMethods.gamma_k(rule, F_z) -> Float64
```

The shared `linesearch!` driver plugs your `γ_k` into the unified descent condition $-\psi(z)^\top d \geq \sigma \alpha \gamma_k \|d\|^2$ and backtracks $\alpha = \rho^i$ for you.

### Example: fractional-power line search

```julia
Base.@kwdef struct LSPower <: AbstractDFLineSearch
    σ::Float64 = 0.01
    ρ::Float64 = 0.6
    p::Float64 = 0.5
end

DFMethods.gamma_k(rule::LSPower, F_z) = norm(F_z)^rule.p
```

`p = 1.0` recovers [`LSII`](@ref); `p = 0.0` recovers [`LSI`](@ref); `p ∈ (0, 1)` interpolates and is often empirically gentler on stiff problems.

## Composing custom components

```julia
using NonlinearSolve

alg = DFProjection(;
    direction  = MPRPL_Direction(),
    linesearch = LSPower(p = 0.5),
    inertial   = Inertial(0.25),           # built-in
    set        = RealSpace(),              # built-in
)

prob = NonlinearProblem((u, p) -> u .- sin.(u), ones(100))
sol  = solve(prob, alg)
sol.retcode    # ReturnCode.Success
```

A runnable version lives at [`examples/extending.jl`](https://github.com/mmogib/DFMethods.jl/blob/main/examples/extending.jl). A DB-backed benchmark comparing it to the paper's default is at `benchmarks/scripts/s05_extending.jl` in the project repository.

## Convergence caveats

### Violating the premises

The framework's convergence theorem (Ibrahim 2026 Thm 3.1) requires:

- a direction with sufficient descent ($-\psi(w)^\top d \geq c\|\psi(w)\|^2$) and boundedness ($\|d\| \leq \bar c \|\psi(w)\|$);
- a line-search $\gamma_k$ bounded below by a positive constant on bounded sets.

If your custom components violate these conditions, the algorithm may still run but you **forfeit the convergence guarantee**. `SpectralThreeTerm` and `MPRPL` both satisfy the direction conditions (proved in their respective papers). `LSPower(p)` with $p > 0$ satisfies the line-search conditions on any region where $\psi$ is bounded away from zero.

### Convergence ≠ convergence rate

Satisfying the convergence theorem's premises buys you *eventual* convergence — not a *rate*. A direction can be theoretically correct yet much slower than another on the same problem, sometimes by orders of magnitude.

**Empirical example.** Running `MPRPL_Direction` + `LSII` against `SpectralThreeTerm` + `LSII` on Dai 2015's two test problems at $n = 1000$, tolerance $10^{-6}$, maxiters 5000 (data from `benchmarks/scripts/s05_extending.jl --all`):

| Problem | Jacobian at solution | SpectralThreeTerm iters | MPRPL iters |
|---|---|---|---|
| **Dai P2:** $F_i(x) = x_i - \sin(x_i)$ | $\mathrm{diag}(1 - \cos x_i) \to 0$ as $x \to 0$ | 9–22 | **stalls at 5000, residual ~5e-5** |
| **Dai P4:** $F(x) = Ax + (e^x - 1)$ | $A + I$ (well-conditioned) | 29–136 | 60–235 ✓ |

On the well-conditioned Dai P4, MPRPL converges fine — 2–3× slower than SpectralThreeTerm but in the same iteration scale. On Dai P2, the Jacobian *vanishes at the solution*, and MPRPL becomes effectively sublinear ($\|\psi_k\| \sim 1/\sqrt{k}$). SpectralThreeTerm survives because its spectral coefficient $\vartheta_k^I = s^\top y / y^\top y$ adapts to local curvature; MPRPL has no such adaptation and reduces to a fixed-point-style iteration on this problem class.

**This is not a bug.** Dai 2015's reported "MPD takes ~12 iterations on Dai P2 at $n = 1000$" is *consistent with this data* — Dai stops at residual ~$3 \times 10^{-4}$ (their `atol + rtol` formula), well above MPRPL's accuracy ceiling on a degenerate-Jacobian problem. Our $10^{-6}$ tolerance is two orders of magnitude tighter than what MPRPL was tested at.

**Takeaway**: validate any custom direction on multiple problem classes — at least one well-conditioned, one ill-conditioned, and one with degenerate Jacobian at the solution — before trusting it on the problem you actually care about. The DB-backed `benchmarks/scripts/s05_extending.jl` is the runnable scaffold for that comparison.

### `Inertial` is not a free no-op

A second finding from the same benchmark: `Inertial(0.25)` accelerated MPRPL on Dai P4 by **3–4×** versus `NoInertial()`:

| MPRPL_LSII on Dai P4, $n = 1000$ | median iters across v1–v6 |
|---|---|
| with `Inertial(0.25)` | 76 |
| with `NoInertial()` | 296 |

So `Inertial(θ)` materially accelerates even a direction (MPRPL) that wasn't designed with inertia in mind. Choose `NoInertial()` only for ablation or when you specifically want to reproduce a paper's inertia-free setup.

## Custom constraint sets

The third extension point — your own constraint set — is covered in [Constraint Sets](@ref). The pattern is:

```julia
struct MyConvexSet <: AbstractConstraintSet
    # … your fields …
end

function DFMethods.project!(y, x, set::MyConvexSet)
    # … compute the orthogonal projection of x onto MyConvexSet, write into y …
    return y
end
```

For a quick one-off, [`UserSet`](@ref) wraps a function `proj!(y, x)` without requiring a new type.
