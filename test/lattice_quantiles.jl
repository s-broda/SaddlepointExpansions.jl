# Lattice variables without conditioning components (tailprob and density with lattice = true,
# the case B empty of Theorem 3 of the conditional paper), against exact probabilities and the
# rows of its table of lattice sums; quantiles of the saddlepoint distribution functions by Newton's
# method in saddlepoint space (quantile, ratio_quantile); normalized densities (normalize = true).

@testset "Lattice, quantiles, normalization" begin
    ppmf(k, l) = k < 0 ? 0.0 : exp(k*log(l) - l - SaddlepointExpansions.loggamma(k + 1.0))
    pupp(k, l) = k <= 0 ? 1.0 : SaddlepointExpansions.gamma_inc(Float64(k), l, 0)[1]   # P(S >= k)
    lgam(x) = SaddlepointExpansions.loggamma(x)

    @testset "lattice sums" begin
        # the table of lattice sums of the paper (lattice_rows.tex): Poisson(5) and the bivariate Poisson with a common
        # component, relative errors of P(S >= k) to min(P, 1 - P) and of P(S = k)
        l1, l2, l0 = 2.0, 2.0, 1.0
        K1 = poisson_cgf(5.0)
        K2 = CGF(t -> l1*(exp(t[1]) - 1) + l2*(exp(t[2]) - 1) + l0*(exp(t[1] + t[2]) - 1), 2)
        nok = 0; ncell = 0
        for (block, cells) in CROWS[0]
            if block == 1
                k = parse(Int, cells[1]); K = K1
                Pex = pupp(k, 5.0); pex = ppmf(k, 5.0)
                r = tailprob(K, k; order=4, lattice=true); f = density(K, k; order=4, lattice=true)
                rc = conditional_tailprob(K, [k], 1; order=4, lattice=true)
                fc = conditional_density(K, [k], 1; order=4, lattice=true)
                @test round(r.tau; digits=2) == pval(cells[3]) + 0.0
            else
                k = parse.(Int, split(cells[1], ",")); K = K2
                Pex = sum(ppmf(y, l0)*pupp(k[1] - y, l1)*pupp(k[2] - y, l2) for y in 0:60)
                pex = sum(ppmf(y, l0)*ppmf(k[1] - y, l1)*ppmf(k[2] - y, l2) for y in 0:minimum(k))
                r = tailprob(K, k; order=4, lattice=true); f = density(K, k; order=4, lattice=true)
                rc = conditional_tailprob(K, k, 1:2; order=4, lattice=true)
                fc = conditional_density(K, k, 1:2; order=4, lattice=true)
                @test round.(r.tau; digits=2) == pval.(split(cells[3], ",")) .+ 0.0
            end
            # the unconditional lattice path is the conditional one with B empty: the same numbers
            @test r.F == rc.F && f.f == fc.f && r.status == "ok"
            @test round(Pex; digits=4) == pval(cells[2]) && round(pex; digits=4) == pval(cells[8])
            comp = vcat((r.F .- Pex) ./ min(Pex, 1 - Pex), f.f ./ pex .- 1)
            ok = rowcheck(comp, vcat(cells[4:7], cells[9:12]), Dict())
            nok += count(ok); ncell += length(ok)
            @test all(ok)
        end
        @test (nok, ncell) == (88, 88)
        # the complement: P(S <= k - 1) + P(S >= k) = 1 at every order, through the reflection
        for k in (3, 8)
            @test maximum(abs.(tailprob(K1, k; order=4, lattice=true).F .+ tailprob(K1, k - 1; order=4, lattice=true, lower=true).F .- 1)) < 1e-14
        end
        # no saddlepoint at the continuity-corrected point below the support
        @test_throws DomainError tailprob(K1, 0; lattice=true)
        r = tailprob(K1, 1; order=4, lattice=true)                   # P(S >= 1) = 1 - P(S = 0)
        @test abs(r.F[4] - pupp(1, 5.0)) < 2e-2*ppmf(0, 5.0)
        @test_throws ArgumentError density(K1, 3; lattice=true, normalize=true)
        # binomial and negative binomial against exact values: the errors fall with the order
        Kb = binomial_cgf(40, 0.3)
        for k in (6, 18, 25)
            Pu = SaddlepointExpansions.beta_inc(Float64(k), 41.0 - k, 0.3)[1]                  # P(S >= k)
            pk = exp(lgam(41.0) - lgam(k + 1.0) - lgam(41.0 - k) + k*log(0.3) + (40 - k)*log(0.7))
            e = abs.(tailprob(Kb, k; order=4, lattice=true).F ./ Pu .- 1)
            ef = abs.(density(Kb, k; order=4, lattice=true).f ./ pk .- 1)
            @test e[4] < 5e-4 && e[4] < e[1] && ef[4] < 1e-4 && ef[4] < ef[1]/10
        end
        Knb = negative_binomial_cgf(5.0, 0.4)                         # failures before the 5th success
        for k in (2, 7, 20)
            Pu = SaddlepointExpansions.beta_inc(Float64(k), 5.0, 0.6)[1]                       # P(S >= k)
            pk = exp(lgam(k + 5.0) - lgam(5.0) - lgam(k + 1.0) + 5log(0.4) + k*log(0.6))
            e = abs.(tailprob(Knb, k; order=4, lattice=true).F ./ Pu .- 1)
            ef = abs.(density(Knb, k; order=4, lattice=true).f ./ pk .- 1)
            @test e[4] < 1e-3 && ef[4] < 1e-4 && ef[4] < ef[1]/10
        end
        # the expansion of the amplitude about the origin (Kolassa) agrees closely for small saddlepoints
        ro = tailprob(K1, 7; order=3, lattice=true, sinh=:origin).F; rs = tailprob(K1, 7; order=3, lattice=true).F
        @test abs(ro[3]/rs[3] - 1) < 1e-4
        # the first-order error of the Poisson tail falls at least like 1/lambda
        e1(lam) = (k = round(Int, lam + 2sqrt(lam)); abs(tailprob(poisson_cgf(lam), k; order=1, lattice=true).F[1]/pupp(k, lam) - 1))
        @test e1(100.0) < e1(25.0)/4
        rb = setprecision(BigFloat, 128) do
            tailprob(K1, big(11); order=4, lattice=true)
        end
        @test eltype(rb.F) === BigFloat && maximum(abs.(Float64.(rb.F) .- tailprob(K1, 11; order=4, lattice=true).F)) < 1e-15
    end

    @testset "quantiles" begin
        Kg = gamma_cgf(3.0)
        for p in (1e-6, 0.05, 0.5, 0.95), lower in (true, false)
            q = quantile(Kg, p; order=4, lower)
            F = lower ? tailprob(Kg, q.q; order=4, lower=true).F[end] : tailprob(Kg, q.q; order=4).F[end]
            @test abs(F - p) <= 1e-12*min(p, 1 - p) && abs(q.F - p) <= 1e-12*min(p, 1 - p)
            ex = lower ? gamma_exact(q.q, 3).cdf : gamma_exact(q.q, 3).upp
            @test abs(ex/p - 1) < 1e-4
            @test saddlepoint(Kg, q.q)[1] ≈ q.tau
        end
        # the normal: the expansion is exact, and so is its quantile
        q = quantile(normal_cgf(1.0, 4.0), 0.025; order=2)
        @test abs(q.q - (1 - 2*sqrt(2)*SaddlepointExpansions.erfcinv(0.05))) < 1e-12
        @test_throws ArgumentError quantile(Kg, 1.5)
        @test_throws ArgumentError quantile(normal_cgf([0.0, 0.0], [1.0 0; 0 1]), 0.5)
        qb = setprecision(BigFloat, 128) do
            quantile(Kg, big"0.05"; order=4, tol=1e-30)
        end
        @test typeof(qb.q) === BigFloat && abs(Float64(qb.q) - quantile(Kg, 0.05; order=4).q) < 1e-12
        # ratios: the noncentral F, against its exact distribution function
        m_, n_, lam = 4, 12, 3
        KF = CGF(t -> ncx2_cgf(m_, lam)([t[1]]) + ncx2_cgf(n_)([t[2]]), 2; domain=t -> t[1] < 0.5 && t[2] < 0.5)
        for p in (0.01, 0.5, 0.99)
            q = ratio_quantile(KF, p; order=4)
            @test abs(ratio_tailprob(KF, q.r; order=4, lower=true).F[end] - p) <= 1e-12*min(p, 1 - p)
            @test abs(dncf_exact_dp(q.r*n_/m_, m_, n_, lam, 0).cdf - p) < 3e-4*min(p, 1 - p)
        end
        q = ratio_quantile(KF, 1e-4; order=4, lower=false)
        @test abs(ratio_tailprob(KF, q.r; order=4).F[end]/1e-4 - 1) < 1e-11
        @test abs(dncf_exact_dp(q.r*n_/m_, m_, n_, lam, 0).upp/1e-4 - 1) < 1e-3
        @test_throws ArgumentError ratio_quantile(Kg, 0.5)
    end

    @testset "normalized densities" begin
        cg(a) = exp(lgam(a) + a + (1/2 - a)*log(a))/sqrt(2pi)       # Daniels's factor for the gamma
        for a in (0.5, 3.0)
            r = density(gamma_cgf(a), 2.0; order=4, normalize=true)
            @test abs(r.info.mass[1]/cg(a) - 1) < (a < 1 ? 1e-8 : 1e-13)
            # the bracket of the gamma expansion is constant, so every normalized order is exact
            @test maximum(abs.(r.f ./ gamma_exact(2.0, a).pdf .- 1)) < (a < 1 ? 1e-8 : 1e-13)
        end
        @test maximum(abs.(density(normal_cgf(1.0, 2.0), 0.3; order=3, normalize=true).info.mass .- 1)) < 1e-13
        @test maximum(abs.(density(normal_cgf([0.0, 1.0], [1.0 0.5; 0.5 2.0]), [0.3, 0.2]; order=2, normalize=true).info.mass .- 1)) < 1e-9
        # the normalized expansion integrates to one, and the masses can be reused
        Kx = ncx2_cgf(4, 3)
        r = density(Kx, 7.0; order=3, normalize=true)
        tot = quadgk(x -> density(Kx, x; order=3, normalize=r.info.mass).f, 1e-9, 10, 40, Inf; rtol=1e-10)[1]
        @test maximum(abs.(tot .- 1)) < 1e-9
        @test density(Kx, 7.0; order=3, normalize=r.info.mass).f == r.f
        # two dimensions: a linear transformation of two independent gamma variables; the mass of
        # the first order is the product of the univariate factors, that of the second order
        # c_1(1 - 1/24 - 1/36), the brackets of the gamma expansion being 1 - 1/(12a)
        Kl = linear_combination(CGF(t -> -2log(1 - t[1]) - 3log(1 - t[2]), 2; domain=t -> t[1] < 1 && t[2] < 1),
                                [1.0 0.5; 0.3 1.0])
        m2 = density(Kl, [1.0, 2.0]; order=2, normalize=true).info.mass
        @test abs(m2[1]/(cg(2.0)*cg(3.0)) - 1) < 1e-9 && abs(m2[2]/(cg(2.0)*cg(3.0)*(1 - 1/24 - 1/36)) - 1) < 1e-9
        # a ratio and a Studentized mean
        m_, n_, lam = 4, 12, 3
        KF = CGF(t -> ncx2_cgf(m_, lam)([t[1]]) + ncx2_cgf(n_)([t[2]]), 2; domain=t -> t[1] < 0.5 && t[2] < 0.5)
        rr = ratio_density(KF, 1.0; order=3, normalize=true)
        tot = quadgk(r -> ratio_density(KF, r; order=3, normalize=rr.info.mass).f, 1e-6, 1, 5, 50, Inf; rtol=1e-9)[1]
        @test maximum(abs.(tot .- 1)) < 1e-8
        mu, nu, n = 1.0, 3.0, 16
        Kn = sample_mean(CGF(t -> mu*t[1] + t[1]^2/2 - nu/2*log(1 - 2t[2]/nu), 2; domain=t -> t[2] < nu/2), n)
        rs = studentized_density(Kn, 1.5; order=2, normalize=true)
        tot = quadgk(q -> studentized_density(Kn, q; order=2, normalize=rs.info.mass).f, -Inf, 0.0, 1.0, 2.5, Inf; rtol=1e-8)[1]
        @test maximum(abs.(tot .- 1)) < 1e-8
        @test abs(rs.info.mass[1] - 1) < 1e-2
    end
end
