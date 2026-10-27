# Studentized means Q = X/sqrt(Y): the expansions of Bao, Broda and Kan against exact values
# (noncentral t) and against the reference values of the Dickey-Fuller application (the Python
# and Julia programs of the paper agree on them to 1e-10).

@testset "Studentized means" begin
    include("dickey_fuller_cgf.jl")
    @testset "noncentral t" begin
        # X ~ N(mu, 1), Y ~ chi2_nu/nu independent, n pairs: Xbar/sqrt(Ybar) = T/sqrt(n), T noncentral t
        mu, nu, n = 1.0, 3.0, 16
        K1 = CGF(t -> mu*t[1] + t[1]^2/2 - nu/2*log(1 - 2t[2]/nu), 2; domain = t -> t[2] < nu/2)
        Kn = sample_mean(K1, n)
        Phi(x) = SaddlepointExpansions.normcdf(x)
        k, th = n*nu/2, 2/(n*nu)
        logg(y) = (k - 1)*log(y) - y/th - SaddlepointExpansions.loggamma(k) - k*log(th)
        exactF(q) = quadgk(y -> exp(logg(y))*Phi(sqrt(n)*(q*sqrt(y) - mu)), 0, Inf; rtol=1e-12)[1]
        exactf(q) = quadgk(y -> exp(logg(y))*sqrt(n*y)*exp(-n*(q*sqrt(y) - mu)^2/2)/sqrt(2pi), 0, Inf; rtol=1e-12)[1]
        for q in (0.5, 1.5, 2.5)
            sd = studentized_saddlepoint(Kn, q)
            @test sd.v > 0 && (q > sd.q0) == (sd.s > 0)
            r = studentized_tailprob(Kn, q; order=3, lower=true)
            ex = exactF(q)
            @test r.status == "ok"
            @test abs(r.info.Fp[1] - ex) < 1e-3
            @test abs(r.F[end] - ex) < 2e-9
            @test abs(r.info.Fp[end] - ex) < abs(r.info.Fp[1] - ex)/100
            ru = studentized_tailprob(Kn, q; order=3)
            @test ru.F[end] ≈ 1 - r.F[end]
            d = studentized_density(Kn, q; order=3)
            fex = exactf(q)
            @test abs(d.f[1]/fex - 1) < 1e-2
            @test abs(d.f[3]/fex - 1) < 1e-6
        end
        # the quantile of the expansion, by the search along the locus
        Q = studentized_quantile(Kn, 0.05; order=3)
        @test abs(studentized_tailprob(Kn, Q.q; order=3, lower=true).F[end] - 0.05) < 1e-9
        @test abs(exactF(Q.q) - 0.05) < 1e-8
        # the locus: the saddlepoint of q lies on it
        sd = studentized_saddlepoint(Kn, 1.5)
        loc = studentized_locus(Kn, sd.s, 0.9*sd.t)
        @test loc.t ≈ sd.t && loc.v ≈ sd.v && loc.q ≈ 1.5
        # BigFloat
        rb = setprecision(BigFloat, 128) do
            studentized_tailprob(Kn, big(1.5); order=3, lower=true)
        end
        @test eltype(rb.F) === BigFloat
        @test abs(Float64(rb.F[end]) - exactF(1.5)) < 2e-9
        @test_throws MethodError studentized_tailprob(K1, [1.0, 2.0]; order=3)
        @test_throws ArgumentError studentized_density(marginal(K1, [1]), 1.0)
    end

    @testset "Dickey-Fuller t-ratio" begin
        # reference values at the 5% asymptotic critical values (c = 0): distribution function with
        # the terms up to n^{-j/2}, j = 0..5, and density with M = 1..6 terms
        ref = Dict(
            1 => (-1.941, [0.04131561, 0.04932910, 0.05021031, 0.05004336, 0.04996497, 0.04997437],
                  [1.137468e-01, 1.145105e-01, 1.142462e-01, 1.143557e-01, 1.145177e-01, 1.133560e-01]),
            2 => (-2.862, [0.04748127, 0.05196366, 0.04837446, 0.04492099, 0.04534934, 0.04962684],
                  [1.288676e-01, 1.235403e-01, 1.262806e-01, 1.274765e-01, 1.212351e-01, 1.265603e-01]),
            3 => (-3.411, [0.04466519, 0.04979224, 0.04990885, 0.04967211, 0.04978553, 0.04999147],
                  [1.307302e-01, 1.305121e-01, 1.304833e-01, 1.306511e-01, 1.304847e-01, 1.305910e-01]))
        for m in 1:3
            q, Fref, fref = ref[m]
            K = dickey_fuller_cgf(m)
            r = studentized_tailprob(K, q; order=3, lower=true)
            @test maximum(abs.(r.info.Fp .- Fref)) < 1e-8
            d = studentized_density(K, q; order=6)          # refined in BigFloat where needed
            @test maximum(abs.(d.f ./ fref .- 1)) < 2e-6
            @test d.status == "ok"
        end
        # exact quantiles of the limit (df_quantiles.json of the paper): the saddlepoint and the
        # distribution function at the 0.75 quantile of M3 under c = -5 (that > 0 there)
        K = dickey_fuller_cgf(3; c=-5.0)
        q = -2.025180818400757
        r = studentized_tailprob(K, q; order=3, lower=true)
        @test r.saddle.s ≈ 2.391156286 atol=1e-8
        @test r.saddle.t ≈ 10.320589682 atol=1e-8
        @test abs(r.info.Fp[end] - 0.739908646) < 1e-8
        # the 5% quantile of the six-term expansion for M2 (exact limit: -2.861569)
        Q = studentized_quantile(dickey_fuller_cgf(2), 0.05; order=3)
        @test Q.q ≈ -2.859305 atol=1e-5
        # the series window of D: the same value on both sides of it
        K2 = dickey_fuller_cgf(2; c=-5.0); K2w = dickey_fuller_cgf(2; c=-5.0, window=1e-3)
        @test K2([0.3, 11.0]) ≈ K2w([0.3, 11.0]) rtol=1e-10
        @test_throws ArgumentError dickey_fuller_cgf(4)
    end
end
