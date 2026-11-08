# Test suite.
#   julia --project=. test/runtests.jl      (or  ] test  in the package environment)
# cgf_interface.jl:   the CGF interface (Taylor-mode derivatives, transformations, tail
#                     probabilities, densities, ratios) against closed-form derivatives and exact
#                     values;
# paper_tables.jl:    a selection of rows of the tables of the paper, recomputed from the CGF;
# reference_matlab.jl: the derivative interface against reference values of the MATLAB programs
#                     (reference_matlab.csv) and the BigFloat paths;
# studentized.jl:     Studentized means (noncentral t against exact values, the Dickey-Fuller
#                     t-ratio against the reference values of the paper);
# conditional.jl:     the conditional expansions against the rows of the tables of the conditional
#                     paper (conditional_rows.txt) and exact values, consistency checks, BigFloat;
# lattice_quantiles.jl: lattice sums (lattice = true), quantiles in saddlepoint space, normalized
#                     densities.
using SaddlepointExpansions, Test, Printf, LinearAlgebra
const quadgk = SaddlepointExpansions.quadgk

reldiff(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1e-300))

@testset "SaddlepointExpansions" begin
    include("cgf_interface.jl")
    include("paper_tables.jl")
    include("reference_matlab.jl")
    include("studentized.jl")
    include("conditional.jl")
    include("lattice_quantiles.jl")
end
