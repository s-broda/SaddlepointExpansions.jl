# Rows of the tables of Broda and Kan recomputed from the cumulant generating function alone.
# The printed relative errors have two significant digits, so the comparison allows 6% of the
# printed value plus an absolute margin for the rounding of the exact values (seven digits).
# (The complete tables are reproduced by the drivers of the replication repository.)

const quadgk = SaddlepointExpansions.quadgk

function matchrow(computed, printed, exact_digits)
    marg = 6e-2 .* abs.(printed) .+ 2*10.0^(-exact_digits)
    return all(abs.(computed .- printed) .<= marg)
end

@testset "rows of the tables of the paper" begin
    @testset "Table 1: Wang's example, n = 10" begin
        A = [1 0; 1 1; 0 1]; n = 10
        K = exponential_sums_cgf(A; n, shift=[-2.0, -2.0])
        gpdf(u) = u <= 0 ? 0.0 : n*gamma_exact(n*u, n).pdf
        gcdf(u) = u <= 0 ? 0.0 : gamma_exact(n*u, n).cdf
        rows = (([-0.5, -0.5], [-1.4e-2, -7.2e-5, 7.4e-6, 2.0e-7, -3.3e-8]),
                ([-1.0, 2.0], [-2.0e-3, -4.9e-5, -1.7e-7, 2.7e-8, 4.5e-10]),
                ([0.0, 0.0], [-3.8e-3, 3.9e-6, 2.0e-6, 4.6e-8, -1.0e-8]),
                ([-1.0, -1.0], [-1.7e-2, -3.2e-4, 8.7e-6, 8.3e-7, -2.6e-8]))
        for (x, printed) in rows
            ex = quadgk(u -> gpdf(u)*gcdf(x[1] + 2 - u)*gcdf(x[2] + 2 - u), 0, min(x...) + 2; rtol=1e-13)[1]
            r = tailprob(K, x; order=5, lower=true)
            @test matchrow((r.F .- ex) ./ ex, printed, 9)
        end
    end
    @testset "Table 3: the example of Kolassa and Li, n = 5" begin
        A = [1 0; 1 1; 0 1]; n = 5
        K = exponential_sums_cgf(A; n)
        rows = (([3.0, 3.0], 2.215196e-2, [-3.3e-2, -3.8e-4, 9.8e-5, -1.5e-6, -2.1e-6]),
                ([3.5, 4.0], 1.655068e-3, [-2.7e-2, -6.3e-4, 9.5e-5, 6.6e-6, -2.8e-6]),
                ([2.5, 3.0], 4.123179e-2, [-1.9e-2, -2.7e-4, 4.2e-5, 4.9e-7, -9.7e-7]))   # tau_1 = 0
        for (x, ex, printed) in rows
            r = tailprob(K, x; order=5)
            @test matchrow((r.F .- ex) ./ ex, printed, 7)
        end
        r = tailprob(K, [2.5, 3.0]; order=2)
        @test r.tau[1] == 0                                     # saddlepoint on the boundary
    end
    @testset "Table 8: the diagonal of a trivariate Wishart matrix, n = 5" begin
        V = 0.75 .* Matrix(1.0I, 3, 3) .+ 0.25
        K = wishart_diag_cgf(V, 5)
        x = [6.5, 6.5, 7.5]
        r = tailprob(K, x; order=4)
        @test matchrow((r.F .- 1.827417e-2) ./ 1.827417e-2, [-2.4e-1, -8.2e-3, 1.2e-3, 4.4e-5], 7)
        f = density(K, x; order=4)
        @test matchrow((f.f .- 5.013887e-4) ./ 5.013887e-4, [1.1e-1, -6.4e-3, -5.5e-4, 1.6e-4], 7)
    end
end
