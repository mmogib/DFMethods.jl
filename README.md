# DFMethods

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mmogib.github.io/DFMethods.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mmogib.github.io/DFMethods.jl/dev/)
[![Build Status](https://github.com/mmogib/DFMethods.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mmogib/DFMethods.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mmogib/DFMethods.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mmogib/DFMethods.jl)

**Derivative-free projection methods for constrained nonlinear equations, integrated with NonlinearSolve.jl.**

Solve

$$\text{Find } u^* \in X \subset \mathbb{R}^n \text{ such that } F(u^*) = 0,$$

where $X$ is closed convex and $F$ is continuous and (pseudo-)monotone. **No derivatives required.**

## Install

```julia
] add DFMethods
```

Full installation options (registry, REPL, unreleased-from-GitHub) are on the [docs home page](https://mmogib.github.io/DFMethods.jl/stable/#Installation).

## Documentation

Full documentation: <https://mmogib.github.io/DFMethods.jl/stable/>.

- [Quickstart](https://mmogib.github.io/DFMethods.jl/stable/quickstart/) — runnable examples for unconstrained, box, and convex-set problems
- [Algorithm](https://mmogib.github.io/DFMethods.jl/stable/algorithm/) — the outer-iteration structure and pluggable components
- [Constraint Sets](https://mmogib.github.io/DFMethods.jl/stable/constraints/) — built-in feasibility sets and `ConstrainedNonlinearProblem`
- [Extending](https://mmogib.github.io/DFMethods.jl/stable/extending/) — define your own search direction, line search, iterate update, …
- [Comparisons](https://mmogib.github.io/DFMethods.jl/stable/comparisons/) — DFMethods.jl vs related Julia packages
- [API Reference](https://mmogib.github.io/DFMethods.jl/stable/api/)

## Citation

The current release's citation metadata is in [`CITATION.cff`](CITATION.cff) — GitHub's *Cite this repository* button reads this file. A formal `@software{}` BibTeX block with the Zenodo DOI is added after each tagged release; see the [GitHub Releases page](https://github.com/mmogib/DFMethods.jl/releases).

If you build on a specific algorithmic component, please additionally cite the originating source — each component's docstring names it with a DOI link.

## License

MIT
