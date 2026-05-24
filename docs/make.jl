using DFMethods
using Documenter
using CommonSolve
using SciMLBase

DocMeta.setdocmeta!(DFMethods, :DocTestSetup, :(using DFMethods); recursive=true)

makedocs(;
    modules=[DFMethods],
    authors="Mohammed Alshahrani <mshahrani@kfupm.edu.sa>",
    sitename="DFMethods.jl",
    # Only exported names need to appear in @docs blocks. Internal helpers
    # (init_state, SolodovSvaiterState, HalpernState, _constraint_set, …)
    # may carry docstrings without being included in the manual.
    checkdocs=:exports,
    format=Documenter.HTML(;
        canonical="https://mmogib.github.io/DFMethods.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home"            => "index.md",
        "Quickstart"      => "quickstart.md",
        "Tutorial"        => "tutorial.md",
        "Algorithm"       => "algorithm.md",
        "Constraint Sets" => "constraints.md",
        "Extending"       => "extending.md",
        "Comparisons"     => "comparisons.md",
        "API Reference"   => "api.md",
    ],
)

deploydocs(;
    repo="github.com/mmogib/DFMethods.jl",
    devbranch="main",
)
