using Documenter, SaddlepointExpansions

makedocs(;
    sitename = "SaddlepointExpansions.jl",
    modules = [SaddlepointExpansions],
    repo = Documenter.Remotes.GitHub("s-broda", "SaddlepointExpansions.jl"),
    authors = "Simon A. Broda and Raymond Kan",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://s-broda.github.io/SaddlepointExpansions.jl",
        mathengine = Documenter.MathJax3(),
        size_threshold_warn = 300*1024,
        size_threshold = 600*1024,
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "Library of cumulant generating functions" => "library.md",
        "The derivative interface" => "lowlevel.md",
        "Background: the expansions" => "background.md",
        "API reference" => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(;
    repo = "github.com/s-broda/SaddlepointExpansions.jl.git",
    devbranch = "main",
    push_preview = false,
)
