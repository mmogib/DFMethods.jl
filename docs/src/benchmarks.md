```@meta
CurrentModule = DFMethods
```

# Benchmarks

This page summarizes the experimental results that motivate the default
configuration of [`DFProjection`](@ref). All data come from the harness in
the [`benchmarks/`](https://github.com/mmogib/DFMethods.jl/tree/main/benchmarks)
directory of the project repository.

## Setup

Experimental design — the `s30_benchmark.jl --all` configuration:

| Factor                   | Levels                                                                        |
|--------------------------|-------------------------------------------------------------------------------|
| Search direction         | `SpectralThreeTerm`, `MPRPL`, `GCGPM`, `GMOPCGM` (4)                          |
| Line search ($\gamma_k$) | `ConstantBacktrack`, `ResidualNormBacktrack`, plus three legacy variants (5)  |
| Test problem             | 10 monotone nonlinear-equation problems (Dai 2015, ELSCAM, Yin 2021)          |
| Initial point            | 5 paper-standard starts (v1–v5)                                               |
| Dimension                | 1 000, 10 000, 15 000 (3)                                                     |
| **Total runs**           | **3 000**                                                                     |

Constraint set throughout: $\Omega(a, b, c) = [a,b]^n \cap \{x : \sum_i x_i
\leq c\}$ (Ibrahim 2026 polyhedral domain), implemented as
[`CappedBox`](@ref). Tolerance $\varepsilon = 10^{-6}$; iteration budget
$5\,000$.

Hardware: single Julia 1.10 process; wall-time 700.8 s (about 11.7 min)
for the complete sweep.

!!! note "Notation reconciliation"
    The benchmark sweep was run under the v0.1 API and stored its method
    labels using the pre-v0.2 names. The mapping to current types is:
    `LSI` → [`ConstantBacktrack`](@ref);
    `LSII` → [`ResidualNormBacktrack`](@ref);
    `LSVII` → [`AdaptiveClampedBacktrack`](@ref).
    The variants `LSIII`, `LSV`, and `LSVI` are not shipped in v0.2 — see
    [Extending](@ref) §1 for the recipe to recover any of them as a
    user-defined line search.

## Headline outcome

Of the 3 000 runs, 2 544 (**84.8 %**) converged within tolerance and the
iteration budget. The dominant separator was the choice of *search
direction*, not the line search: `SpectralThreeTerm` averaged 20–26
outer iterations across line-search variants, while every other direction
averaged 700–900. The shipped default
`DFProjection(direction = SpectralThreeTerm(), linesearch =
ResidualNormBacktrack(), ...)` corresponds to the
`SpectralThreeTerm` + `ResidualNormBacktrack` cell of the table below.

## Per-configuration aggregates

Averaged over all problems, initial points, and dimensions (150 runs each).
"Avg iter" and "Avg F-evals" exclude divergent runs; "Total time" is the
sum over all 150 runs.

| Direction               | Line search                  | Convergence | Avg iter | Avg F-evals | Total time |
|-------------------------|-------------------------------|------------:|---------:|------------:|-----------:|
| `SpectralThreeTerm`     | `ConstantBacktrack`           | 86.0 %      |   210.0  |   2 761.1   |  32.1 s    |
| `SpectralThreeTerm`     | **`ResidualNormBacktrack`**   | **86.0 %**  |   **20.5** | **77.2** |  **1.1 s** |
| `SpectralThreeTerm`     | LS III (legacy)               | 86.0 %      |    26.2  |     106.8   |   1.7 s    |
| `SpectralThreeTerm`     | LS V (legacy)                 | 86.0 %      |    26.3  |     107.1   |   1.7 s    |
| `SpectralThreeTerm`     | LS VI (legacy)                | 86.0 %      |    22.8  |      87.4   |   1.3 s    |
| MPRPL                   | `ConstantBacktrack`           | 84.0 %      |   856.8  |   2 110.8   |  34.8 s    |
| MPRPL                   | `ResidualNormBacktrack`       | 82.7 %      |   892.5  |   5 688.1   |  63.9 s    |
| GCGPM                   | `ConstantBacktrack`           | 85.3 %      |   822.1  |   1 995.6   |  35.0 s    |
| GCGPM                   | `ResidualNormBacktrack`       | 85.3 %      |   838.3  |   4 372.0   |  57.6 s    |
| GMOPCGM                 | `ConstantBacktrack`           | 84.0 %      |   737.8  |   1 933.2   |  31.2 s    |
| GMOPCGM                 | `ResidualNormBacktrack`       | 84.7 %      |   736.3  |   2 717.8   |  55.0 s    |

(Legacy line-search rows for the non-`SpectralThreeTerm` directions
omitted for brevity; the full table is available by running
`s30_benchmark.jl --summary` against the committed
`benchmarks/results/experiments.db`.)

## Reading the table

Three patterns stand out.

1. **Direction dominates iteration count.**
   The `SpectralThreeTerm` direction converges in 20–26 outer iterations
   on average. The PRP-style directions (`MPRPL`, `GCGPM`, `GMOPCGM`)
   converge in 700–900. The difference is not a tuning artifact — it is
   driven by `SpectralThreeTerm`'s adaptive spectral coefficient, which
   PRP-type methods lack.

2. **Line search dominates F-evaluation count.**
   Within `SpectralThreeTerm`, `ResidualNormBacktrack` is the clear
   winner: 77 F-evaluations on average versus 2 761 for
   `ConstantBacktrack` and 87–107 for the legacy variants. The line search
   does *not* materially change the iteration count, only the work spent
   inside each iteration.

3. **Convergence rate is essentially flat.**
   Every direction × line-search combination reaches 82–86 % convergence.
   Failures concentrate on a small number of pathological problem ×
   initial-point combinations (notably problems with a vanishing Jacobian
   at the solution; see [Extending](@ref) §8). No combination "rescues"
   these problems.

## Default-configuration rationale

The defaults are chosen to optimize against the per-iteration F-evaluation
count and the wall-clock time, subject to no degradation in convergence
rate:

```julia
DFProjection(;
    direction      = SpectralThreeTerm(),       # 20–26 vs 700+ avg iters
    linesearch     = ResidualNormBacktrack(),   # 77 vs 2761 avg F-evals
    inertial       = Inertial(0.25),            # 3–4× speedup on MPRPL-class
    iterate_update = SolodovSvaiterProjection(),# reproduces Ibrahim 2026
    abstol         = 1e-6,
    maxiters       = 2000,
)
```

These are the empirical settings; the convergence theorem of Ibrahim 2026
holds for any direction × line search × iterate update obeying the
contracts of [Extending](@ref).

## Reproducing

The full sweep is reproducible from a clean checkout. From the project
root:

```bash
cd benchmarks/
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/s30_benchmark.jl --all --force   # ~12 min
julia --project=. scripts/s30_benchmark.jl --summary       # tabulate from DB
```

The `--force` flag re-runs every cell; omit it to resume an interrupted
sweep. Results are appended to `benchmarks/results/experiments.db` (SQLite,
content-addressable per configuration hash) — see the harness's own
`CLAUDE.md` for schema and CLI details.

A focused performance-profile script
(`scripts/s70_figures.jl`) generates Dolan–Moré profiles for each
direction–line-search combination across the iteration, F-evaluation, and
CPU-time metrics. Static figures are not yet checked into the docs; their
addition is tracked as a follow-up.
