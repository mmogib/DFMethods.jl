using DFMethods
using Documenter
using CommonSolve
using SciMLBase

DocMeta.setdocmeta!(DFMethods, :DocTestSetup, :(using DFMethods); recursive=true)

makedocs(;
    modules=[DFMethods],
    authors="Mohammed Alshahrani <mshahrani@kfupm.edu.sa>",
    sitename="DFMethods.jl",
    format=Documenter.HTML(;
        canonical="https://mmogib.github.io/DFMethods.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home"            => "index.md",
        "Quickstart"      => "quickstart.md",
        "Algorithm"       => "algorithm.md",
        "Constraint Sets" => "constraints.md",
        "Extending"       => "extending.md",
        "API Reference"   => "api.md",
    ],
)

deploydocs(;
    repo="github.com/mmogib/DFMethods.jl",
    devbranch="main",
)
