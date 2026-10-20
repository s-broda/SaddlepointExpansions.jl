# Test suite.
#   julia --project=. test/runtests.jl      (or  ] test  in the package environment)
# cgf_interface.jl:   the CGF interface (Taylor-mode derivatives, transformations, tail
#                     probabilities, densities, ratios) against closed-form derivatives and exact
#                     values;
# paper_tables.jl:    a selection of rows of the tables of the paper, recomputed from the CGF;
# reference_matlab.jl: the derivative interface against reference values of the MATLAB programs
#                     (reference_matlab.csv) and the BigFloat paths.
using SaddlepointExpansions, Test, Printf, LinearAlgebra
const quadgk = SaddlepointExpansions.quadgk

reldiff(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1e-300))

@testset "SaddlepointExpansions" begin
    include("cgf_interface.jl")
    include("paper_tables.jl")
    include("reference_matlab.jl")
end
