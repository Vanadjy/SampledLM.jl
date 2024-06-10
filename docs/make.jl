using SampledLM
using Documenter

DocMeta.setdocmeta!(SampledLM, :DocTestSetup, :(using SampledLM); recursive = true)

makedocs(;
  modules = [SampledLM],
  doctest = true,
  linkcheck = false,
  strict = false,
  authors = "Valentin Dijon <valentin.dijon@polymtl.ca>, Youssef Diouane <youssef.diouane@pooymtl.ca> and Dominique Orban <dominique.orban@polymtl.ca>",
  repo = "https://github.com/JuliaSmoothOptimizers/SampledLM.jl/blob/{commit}{path}#{line}",
  sitename = "SampledLM.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSmoothOptimizers.github.io/SampledLM.jl",
    assets = ["assets/style.css"],
  ),
  pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo = "github.com/JuliaSmoothOptimizers/SampledLM.jl",
  push_preview = true,
  devbranch = "main",
)
