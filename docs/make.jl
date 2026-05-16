using DFMethods
using Documenter

DocMeta.setdocmeta!(DFMethods, :DocTestSetup, :(using DFMethods); recursive=true)

makedocs(;
    modules=[DFMethods],
    authors="Mohammed Alshahrani <mshahrani@kfupm.edu.sa>",
    sitename="DFMethods.jl",
    format=Documenter.HTML(;
        canonical="https://mmogib.github.io/DFMethods.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mmogib/DFMethods.jl",
    devbranch="master",
)
