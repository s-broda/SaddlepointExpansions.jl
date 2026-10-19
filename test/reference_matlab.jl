# The derivative interface against reference values of the MATLAB programs of the paper
# (reference_matlab.csv, written by make_reference.m of the replication repository), and the
# BigFloat paths.

const REF = let d = Dict{String,Vector{Float64}}()
    for (i, line) in enumerate(eachline(joinpath(@__DIR__, "reference_matlab.csv")))
        i == 1 && continue
        nm, k, v = split(line, ',')
        push!(get!(d, nm, Float64[]), parse(Float64, v))
    end
    d
end
function check(nm, v, tol)
    v = collect(v)
    d = reldiff(v, REF[nm][1:length(v)])
    d > tol && @printf("  %-28s max rel. diff %.1e (tol %.0e)\n", nm, d, tol)
    @test d <= tol
    return d
end

@testset "derivative interface vs MATLAB reference values" begin
    @testset "spa_uni (gamma)" begin
        a = 3; order = 6
        for x in (0.5, 1.5, 2.9, 3.0, 4.5, 9.0, 20.0)
            s = 1 - a/x
            Kd = [k == 0 ? -a*log(1 - s) : a*factorial(k - 1)/(1 - s)^k for k in 0:2order+1]
            r = spa_uni(Kd, s, x, order)
            nm = "uni_gamma_x" * (isinteger(x) ? string(Int(x)) : string(x))
            check(nm * "_pdf", r.pdf, 1e-11); check(nm * "_cdf", r.cdf, 1e-11)
            check(nm * "_upp", r.upp, 1e-11); check(nm * "_kappa", r.kappa, 1e-11)
        end
    end
    @testset "noncentral chi-squared and F" begin
        for (i, p) in enumerate(([30, 5, 10], [2, 5, 10], [400, 50, 300]))
            r = ncx2_spa(p..., 6)
            check("ncx2_$(i)_cdf", r.cdf, 1e-11); check("ncx2_$(i)_pdf", r.pdf, 1e-11)
            check("ncx2_$(i)_upp", r.upp, 1e-12)
        end
        for (i, p) in enumerate(([2.5, 4, 12, 3], [0.3, 10, 20, 50], [5, 3, 7, 100]))
            r = ncf_spa(p..., 6)
            check("ncf_$(i)_cdf", r.cdf, 1e-12); check("ncf_$(i)_pdf", r.pdf, 1e-12)
            check("ncf_$(i)_upp", r.upp, 1e-12)
        end
        for (i, p) in enumerate(([2.5, 4, 12, 3, 7], [0.3, 10, 20, 50, 1], [5, 3, 7, 100, 1000],
                                 [1.1, 1, 1, 1, 1]))
            r = dncf_spa(p..., 6)
            check("dncf_$(i)_cdf", r.cdf, 1e-12); check("dncf_$(i)_pdf", r.pdf, 1e-12)
            check("dncf_$(i)_upp", r.upp, 1e-12)
            ex = dncf_exact_dp(p...)
            check("dncfex_$i", [ex.cdf, ex.upp, ex.pdf], 1e-12)
        end
    end
    @testset "bivariate (pairs of quadratic forms)" begin
        # orders with cancellation indicator kappa are compared to a tolerance kappa*1e-14
        mm = [15, 20, 10]; ll = [5, 10, 8]
        Kf = (nu, t) -> bqfcc_Kfun(nu, t, mm, ll)
        X = ([55.0, 70.0], [43.0497524691810, 53.7445626465380], [90.0, 110.0], [30.0, 70.0])
        for (i, x) in enumerate(X)
            r = spa_bi(Kf, x, 5)
            check("bi_$(i)_tau", r.tau, 1e-12)
            check("bi_$(i)_pdf", r.pdf[1:4], 1e-11)
            if all(r.tau .>= 0)
                d = abs.(r.Fup .- REF["bi_$(i)_Fup"]) ./ REF["bi_$(i)_Fup"]
                @test all(d .<= max.(1e-11, 1e-14 .* REF["bi_$(i)_kappa"]))
            end
            o = spa_orthant2(Kf, x, 5)
            d = abs.(o.Fup .- REF["bi_$(i)_orth"]) ./ REF["bi_$(i)_orth"]
            kap = all(r.tau .>= 0) ? REF["bi_$(i)_kappa"] : fill(1.0, 5)
            @test all(d .<= max.(1e-11, 1e-14 .* kap))
        end
    end
    @testset "quadratic forms" begin
        A = reshape(REF["qf_A"], 6, 6); B = reshape(REF["qf_B"], 6, 6); C = reshape(REF["qf_C"], 6, 6)
        K = qf_Kmulti([0.3, -0.2, 0.1], [A, B, C], 6)
        for key in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 1, 0), (1, 1, 1), (3, 0, 3), (0, 6, 0),
                    (2, 2, 2), (1, 4, 1))
            check("qf_" * join(key, "_"), [K[key]], 1e-12)
        end
        Kb = setprecision(BigFloat, 200) do
            qf_Kmulti(big.([0.3, -0.2, 0.1]), [big.(A), big.(B), big.(C)], 6)
        end
        @test abs(Float64(Kb[(2, 2, 2)]) - K[(2, 2, 2)]) <= 1e-12*abs(K[(2, 2, 2)])
    end
    @testset "three and four dimensions" begin
        m3 = [15, 20, 25, 10]; l3 = [5, 10, 0, 8]
        Kf3 = (nu, t) -> mvcc_Kfun(nu, t, m3, l3)
        ex = mvcc_exact([80.0, 95.0, 100.0], m3, l3)
        check("tri_exact", [ex.Fup, ex.pdf], 1e-10)
        r = spa_tri(Kf3, [80.0, 95.0, 100.0], 3; refine=false)
        check("tri_tau", r.tau, 1e-12); check("tri_pdf", r.pdf, 1e-11)
        d = abs.(r.Fup .- REF["tri_Fup"]) ./ REF["tri_Fup"]
        @test all(d .<= max.(1e-11, 1e-14 .* REF["tri_kappa"]))
        r4 = spa_tri(Kf3, [80.0, 95.0, 100.0], 4)          # whitened: no refinement needed
        @test r4.refined == (REF["tri_refined4"][1] == 1) && r4.info.status == "ok"
        check("tri_Fup4", r4.Fup, 1e-11); check("tri_kappa4", r4.kappa, 1e-8)
        @test all(r4.err .<= 1e-6 .* r4.Fup)               # change with half the nodes
        o = spa_orthant(Kf3, [55.0, 70.0, 75.0], 3)
        check("tri_orth", o.Fup, 1e-10)
        ex = mvcc_exact([55.0, 70.0, 75.0], m3, l3)
        check("tri_orth_exact", [ex.Fup], 1e-10)
        m4 = [15, 20, 25, 30, 10]; l4 = [5, 10, 0, 3, 8]
        Kf4 = (nu, t) -> mvcc_Kfun(nu, t, m4, l4)
        x4 = [70.0, 85.0, 95.0, 105.0]
        ex = mvcc_exact(x4, m4, l4)
        check("mv4_exact", [ex.Fup, ex.pdf], 1e-10)
        r = spa_mv(Kf4, x4, 3)
        check("mv4_pdf", r.pdf, 1e-12)
        @test r.refined == (REF["mv4_refined"][1] == 1)
        @test r.info.status == "negative saddlepoint component (use spa_orthant)" && all(isnan, r.Fup)
        # orthant probability by quasi-Monte Carlo: the differences from the exact value at
        # orders two and three (relative errors 1e-4 and 5e-6) are within a few standard errors
        o = spa_orthant(Kf4, x4, 3)
        @test abs(o.Fup[3] - ex.Fup) <= 1e-4*ex.Fup && all(o.err .<= 1e-4*ex.Fup)
        # d = 2 through spa_mv: density identical to spa_bi
        mm = [15, 20, 10]; ll = [5, 10, 8]
        rb = spa_bi((nu, t) -> bqfcc_Kfun(nu, t, mm, ll), [55.0, 70.0], 4)
        rm = spa_mv((nu, t) -> mvcc_Kfun(nu, t, mm, ll), [55.0, 70.0], 4; method=:none)
        @test reldiff(rm.pdf, rb.pdf) < 1e-12
    end
    @testset "whitened evaluation and options" begin
        # spa_saddle, direct evaluation at a negative saddlepoint, logscale
        A = [1 0; 1 1; 0 1]; K5 = (nu, t) -> lin_exp_Kfun(nu, t, A, 5)
        tau, si = spa_saddle(K5, [2.5, 4.0])
        check("sad_tau", tau, 1e-12); @test si.decrement <= 1e-12
        check("bi_direct", spa_bi(K5, [2.5, 4.0], 3; direct=true).Fup, 1e-10)
        check("bi_logscale", spa_bi(K5, [3.0, 3.5], 3; logscale=-2).Fup, 1e-11)
        # Wishart diagonal (Table 8 model)
        V = 0.75 .* [1.0 0 0; 0 1 0; 0 0 1] .+ 0.25
        Kw = wishdiag_kfun(V, 5)
        r = spa_tri(Kw, [6.5, 6.5, 7.5], 4)
        check("wish_tau", r.tau, 1e-12); check("wish_pdf", r.pdf, 1e-11)
        check("wish_Fup", r.Fup, 1e-11)
        # approximation of Kolassa and Li (finite differences: agreement to rounding of v'')
        check("kl_bivar", [kl_bivar(K5, [3.0, 3.5])[1], kl_bivar(K5, [3.0, 3.5]; ord=(2, 1))[1]], 1e-6)
        # approximation of Wang (1990) against the values he prints (Table 1, n = 10; 4 decimals)
        K10 = (nu, t) -> lin_exp_Kfun(nu, t, A, 10, [-2.0, -2.0])
        for (x, y, w) in ((-0.5, 2.0, 0.129), (0.0, 1.5, 0.537), (1.0, 1.0, 0.958), (-0.8, -0.8, 0.0032), (2.0, 2.0, 0.99955))
            @test abs(wang_bivar(K10, [x, y])[1] - w) <= 0.6e-3*(w < 0.01 ? 0.1 : 1)
        end
        # input validation and status reporting
        @test_throws ArgumentError spa_bi(K5, [1.0, 2.0, 3.0], 2)
        @test_throws ArgumentError spa_mv(K5, [3.0, 3.5], 0)
        @test_throws ArgumentError ncf_spa(-1.0, 3, 4, 1, 2)
        m3 = [15, 20, 25, 10]; l3 = [5, 10, 0, 8]
        Kf3 = (nu, t) -> mvcc_Kfun(nu, t, m3, l3)
        re = setprecision(BigFloat, 128) do
            spa_tri(Kf3, big.([80.0, 95.0, 100.0]), 2)
        end
        @test re.info.method_cdf === :exact && re.info.status == "no error estimate" && all(isnan, re.err)
    end
    @testset "variable precision" begin
        # univariate: BigFloat recursion agrees with the Float64 Gauss-Hermite branch (|u| > 2.5)
        a = 3; x = 20.0; s = 1 - a/x; order = 6
        Kd = [k == 0 ? -a*log(1 - s) : a*factorial(k - 1)/(1 - s)^k for k in 0:2order+1]
        rd = spa_uni(Kd, s, x, order)
        rb = setprecision(BigFloat, 200) do
            sb = 1 - big(a)/big(x)
            Kb = [k == 0 ? -a*log(1 - sb) : a*factorial(big(k - 1))/(1 - sb)^k for k in 0:2order+1]
            spa_uni(Kb, sb, big(x), order)
        end
        @test reldiff(Float64.(rb.upp), rd.upp) < 1e-11
        # bivariate: exact I(nu) in BigFloat against the quadrature in double precision
        mm = [15, 20, 10]; ll = [5, 10, 8]
        Kf = (nu, t) -> bqfcc_Kfun(nu, t, mm, ll)
        for x in ([55.0, 70.0], [90.0, 110.0])
            rd = spa_bi(Kf, x, 4)
            rb = setprecision(BigFloat, 200) do
                spa_bi(Kf, big.(x), 4)
            end
            @test reldiff(Float64.(rb.tau), rd.tau) < 1e-14
            @test reldiff(Float64.(rb.pdf[1:4]), rd.pdf[1:4]) < 1e-11
            d = abs.(Float64.(rb.Fup) .- rd.Fup) ./ Float64.(rb.Fup)
            @test all(d .<= max.(1e-12, 1e-14 .* rd.kappa))
        end
    end
    @testset "review of 15 October 2026" begin
        normk(nu, t, S) = (k = sum(nu); k == 0 ? dot(t, S*t)/2 : k == 1 ? dot(S*t, collect(nu)) :
            k == 2 ? (ii = findall(>(0), collect(nu)); length(ii) == 1 ? S[ii[1], ii[1]] : S[ii[1], ii[2]]) : zero(eltype(t)))
        function lgmix(nu, t, A, a)                      # X = A U, U_j iid Gamma(a,1); t*A of K(t)
            z = 1 .- vec(t'*A)
            any(z .<= 0) && return NaN
            k = sum(nu)
            k == 0 && return -a*sum(log.(z))
            return a*factorial(k - 1)*sum(reduce((u, v) -> u .* v, (A[i, :] .^ nu[i] for i in eachindex(nu))) ./ z .^ k)
        end
        gamnorm(nu, t, a, b) = (1 - b*t[1] <= 0 ? NaN : (k = sum(nu); nz = findall(>(0), collect(nu));
            k == 0 ? -a*log(1 - b*t[1]) + sum(abs2, t[2:end])/2 : length(nz) > 1 ? 0.0 :
            nz[1] == 1 ? a*factorial(k - 1)*b^k/(1 - b*t[1])^k : k == 1 ? t[nz[1]] : k == 2 ? 1.0 : 0.0))
        # Newton's method is invariant under changes of units
        for b in (1.0, 1e8, 1e12, 1e100)
            s, _ = spa_saddle((nu, t) -> gamnorm(nu, t, 10, b), [15b, 0.0])
            @test abs(s[1]*b*3 - 1) < 1e-14 && s[2] == 0
        end
        # density prefactor in the log domain
        for v in (1e-160, 1e160)
            @test abs(spa_bi((nu, t) -> normk(nu, t, v*Matrix(1.0I, 2, 2)), [0.0, 0.0], 1).pdf[1]*(2pi*v) - 1) < 1e-13
            p = spa_mv((nu, t) -> normk(nu, t, v*Matrix(1.0I, 3, 3)), zeros(3), 1; refine=false).pdf[1]
            @test abs(p*(2pi*v)^1.5 - 1) < 1e-13
        end
        # whitened coefficients: X = (U + e V, U - e V); p3/p1 = 1 - 1/60 + 1/7200
        A = [1 1e-3; 1 -1e-3]
        K = (nu, t) -> lgmix(nu, t, A, 10)
        x = [K((1, 0), [0.0, 0.0]), K((0, 1), [0.0, 0.0])]
        r = spa_bi(K, x, 3)
        @test occursin("coefficients inaccurate", r.status) && r.coef_err_pdf[3] > 1e-3
        r = spa_bi(K, x, 3; Kwhite=(nu, u, T) -> lgmix(nu, u, T'*A, 10))
        @test abs(r.pdf[3]/r.pdf[1] - (1 - 1/60 + 1/7200)) < 1e-9
        # inclusion-exclusion: Kwhite transformed; logscale
        K = (nu, t) -> lgmix(nu, t, Matrix(1.0I, 3, 3), 10)
        F1 = spa_orthant(K, [5.0, 15.0, 15.0], 2; refine=false).Fup
        F2 = spa_orthant(K, [5.0, 15.0, 15.0], 2; refine=false, Kwhite=(nu, u, T) -> lgmix(nu, u, Matrix(T'), 10)).Fup
        @test all(abs.(F2 .- F1) .<= 1e-14 .* abs.(F1))
        K = (nu, t) -> normk(nu, t, Matrix(1.0I, 2, 2))
        F0 = spa_orthant2(K, [-1.0, 1.0], 1).Fup[1]
        @test abs(spa_orthant2(K, [-1.0, 1.0], 1; logscale=1).Fup[1]/(F0/exp(1)) - 1) < 1e-14
        # normal orthant probabilities with correlations near +-1
        for rho in (0.999, 0.999999, -0.999, -0.999999)
            S = [1 rho; rho 1]
            ex = 1/4 + asin(rho)/(2pi)
            @test abs(spa_bi((nu, t) -> normk(nu, t, S), [0.0, 0.0], 1).Fup[1]/ex - 1) < 1e-10
            r = spa_tri((nu, t) -> normk(nu, t, [S zeros(2); 0 0 1]), zeros(3), 1)
            @test abs(r.Fup[1]/(ex/2) - 1) < 1e-10 && r.info.status == "ok"
        end
        # substituted coefficients detected and refined in BigFloat (three dimensions)
        A3 = [1 1e-3 0; 1 -1e-3 0; 0 0 1]
        K = (nu, t) -> lgmix(nu, t, A3, 10)
        x = [K((1, 0, 0), zeros(3)), K((0, 1, 0), zeros(3)), K((0, 0, 1), zeros(3))]
        r = spa_mv(K, x, 3; method=:none)
        @test r.info.refined_pdf && maximum(abs.(r.pdf ./ r.pdf[1] .- [1, 0.975, 0.9753125])) < 1e-12
    end
end
