```@meta
CurrentModule = DFMethods
```

# References

The components of DFMethods.jl draw on a body of literature spanning
several decades. This page collects the originating sources, grouped by
the component they relate to.

## Hyperplane projection scheme

Used by [`SolodovSvaiterProjection`](@ref).

> Solodov, M. V., & Svaiter, B. F. (1999). *A New Projection Method for
> Variational Inequality Problems.* SIAM Journal on Control and
> Optimization **37**(3): 765–776.
> [doi:10.1137/S0363012997317475](https://doi.org/10.1137/S0363012997317475).

## Halpern anchoring iteration

Used by [`HalpernUpdate`](@ref).

> Halpern, B. (1967). *Fixed Points of Nonexpanding Maps.* Bulletin of the
> American Mathematical Society **73**(6): 957–961.
> [doi:10.1090/S0002-9904-1967-11864-0](https://doi.org/10.1090/S0002-9904-1967-11864-0).

## Inertial extrapolation

Used by [`Inertial`](@ref) and the inertial step of the outer iteration.

> Alvarez, F., & Attouch, H. (2001). *An Inertial Proximal Method for
> Maximal Monotone Operators via Discretization of a Nonlinear Oscillator
> with Damping.* Set-Valued Analysis **9**(1–2): 3–11.
> [doi:10.1023/A:1011253113155](https://doi.org/10.1023/A:1011253113155).

> Maingé, P.-É. (2008). *Convergence Theorems for Inertial KM-Type
> Algorithms.* Journal of Computational and Applied Mathematics **219**(1):
> 223–236.
> [doi:10.1016/j.cam.2007.07.021](https://doi.org/10.1016/j.cam.2007.07.021).

## Derivative-free spectral-residual lineage

Ecosystem context for derivative-free methods of this class.

> La Cruz, W., Martínez, J. M., & Raydan, M. (2006). *Spectral Residual
> Method without Gradient Information for Solving Large-Scale Nonlinear
> Systems of Equations.* Mathematics of Computation **75**(255):
> 1429–1448.
> [doi:10.1090/S0025-5718-06-01840-0](https://doi.org/10.1090/S0025-5718-06-01840-0).

## Spectral three-term direction

Source of the [`SpectralThreeTerm`](@ref) direction.

> Ibrahim, A. H., Alshahrani, M., & Al-Homidan, S. (2023). *Two Classes
> of Spectral Three-Term Derivative-Free Method for Solving Nonlinear
> Equations with Application.* Numerical Algorithms.
> [doi:10.1007/s11075-023-01679-7](https://doi.org/10.1007/s11075-023-01679-7).

## MPRPL direction (extending example)

Used in the canonical custom-direction example in
`examples/extending.jl`.

> Dai, Z., Chen, X., & Wen, F. (2015). *A Modified Perry's Conjugate
> Gradient Method-Based Derivative-Free Method for Solving Large-Scale
> Nonlinear Monotone Equations.* Applied Mathematics and Computation
> **270**: 378–386.
> [doi:10.1016/j.amc.2015.08.014](https://doi.org/10.1016/j.amc.2015.08.014).

## Unified convergence analysis

A representative convergence theorem covering many configurations of the
framework appears in:

> Ibrahim, A. H., Alshahrani, M., & Al-Homidan, S. (2026). *A Unified
> Derivative-Free Projection Framework for Convex-Constrained Nonlinear
> Equations.* Journal of Optimization Theory and Applications **208**:11.
> [doi:10.1007/s10957-025-02826-x](https://doi.org/10.1007/s10957-025-02826-x).

The benchmark suite under [`Benchmarks`](@ref) reproduces the
experimental setup of this paper.
