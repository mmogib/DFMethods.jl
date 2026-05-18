```@meta
CurrentModule = DFMethods
```

# References

## Foundational algorithm

[Ibrahim2026]
Ibrahim, A. H., Alshahrani, M., & Al-Homidan, S. (2026).
*A Unified Derivative-Free Projection Framework for Convex-Constrained
Nonlinear Equations.*
Journal of Optimization Theory and Applications **208**:11.
[doi:10.1007/s10957-025-02826-x](https://doi.org/10.1007/s10957-025-02826-x).

The package implements Algorithm 1 of this paper as a configurable framework.
[`SpectralThreeTerm`](@ref) reproduces the search-direction rule from the
paper; [`ConstantBacktrack`](@ref), [`ResidualNormBacktrack`](@ref), and
[`AdaptiveClampedBacktrack`](@ref) generalize the seven line-search variants
LS I–LS VII of §2.1; [`SolodovSvaiterProjection`](@ref) is the
hyperplane-projection iterate update of §3.

Other papers cited in the documentation (Dai et al. 2015, Abubakar et al.
2022, …) appear in context — within docstrings or the convergence-caveats
section of [Extending](@ref) — and are not collected here.
