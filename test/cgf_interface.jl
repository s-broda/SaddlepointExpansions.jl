# The CGF interface: derivatives by Taylor-mode differentiation against closed forms, the
# transformations, and tail probabilities, densities and ratios against the derivative interface
# and against exact values.

@testset "CGF interface" begin
    A = [1.0 0; 1 1; 0 1]; n = 10; l = [-2.0, -2.0]
    K = exponential_sums_cgf(A; n, shift=l)                     # Wang's example
    Kf = (nu, t) -> lin_exp_Kfun(nu, t, A, n, l)

    @testset "derivatives" begin
        t = [0.3, -0.2]
        D = derivatives(K, t, 7)
        @test length(D) == binomial(7 + 2, 2)
        @test all(abs(D[nu] - Kf(nu, t)) <= 1e-13*max(1, abs(D[nu])) for nu in keys(D))
        @test derivatives(K, [20.0, 0.0], 3) === nothing          # outside the domain
        Db = setprecision(BigFloat, 200) do
            derivatives(K, big.(t), 7)
        end
        @test eltype(values(Db)) === BigFloat
        @test all(abs(Float64(Db[nu]) - D[nu]) <= 1e-14*max(1, abs(D[nu])) for nu in keys(D))
        # a function that throws outside its domain, without a domain predicate
        Kg = CGF(t -> -3*log(1 - t[1]), 1)
        @test derivatives(Kg, 2.0, 3) === nothing
        @test derivatives(Kg, 0.5, 3)[(3,)] ≈ 3*2/0.5^3
        # normal: cumulants of order three and higher vanish
        Dn = derivatives(normal_cgf([1.0, -1.0], [2.0 0.5; 0.5 1.0]), [0.3, 0.1], 5)
        @test Dn[(1, 0)] ≈ 1 + 2*0.3 + 0.5*0.1 && Dn[(1, 1)] ≈ 0.5
        @test all(abs(Dn[nu]) < 1e-15 for nu in keys(Dn) if sum(nu) >= 3)
        @test_throws ArgumentError derivatives(K, [0.1], 3)
    end

    @testset "transformations" begin
        t = [0.2, 0.1]
        Kr = reflect(K, [1, -1]); Dr = derivatives(Kr, t, 4)
        @test all(Dr[nu] ≈ (-1)^nu[2]*Kf(nu, [t[1], -t[2]]) for nu in keys(Dr))
        Kn = reflect(K); @test derivatives(Kn, t, 1)[(1, 0)] ≈ -Kf((1, 0), -t)
        Km = marginal(K, [2]); Dm = derivatives(Km, [0.4], 5)
        @test all(Dm[(k,)] ≈ Kf((0, k), [0.0, 0.4]) for k in 0:5)
        B = [1.0 2.0; 0.5 -1.0]
        Kl = linear_combination(K, B); Dl = derivatives(Kl, [0.1, -0.05], 1)
        @test Dl[(0, 0)] ≈ Kf((0, 0), B*[0.1, -0.05]) && Dl[(1, 0)] ≈ sum(B[:, 1] .* [Kf((1, 0), B*[0.1, -0.05]), Kf((0, 1), B*[0.1, -0.05])])
        Kg = gamma_cgf(3.0, 2.0); Ks = sample_mean(Kg, 4)
        @test derivatives(Ks, 0.3, 3)[(3,)] ≈ derivatives(Kg, 0.3/4, 3)[(3,)]/4^2
        Ksh = shift(Kg, 1.5)
        @test derivatives(Ksh, 0.2, 2)[(1,)] ≈ derivatives(Kg, 0.2, 2)[(1,)] + 1.5
        @test derivatives(Ksh, 0.2, 2)[(2,)] ≈ derivatives(Kg, 0.2, 2)[(2,)]
        Kc = common_component_cgf([15, 20, 10], [5, 10, 8])
        Ksum = marginal(Kc, [1]) + ncx2_cgf(3, 1)                 # independent sum
        @test derivatives(Ksum, 0.1, 2)[(2,)] ≈ derivatives(marginal(Kc, [1]), 0.1, 2)[(2,)] + derivatives(ncx2_cgf(3, 1), 0.1, 2)[(2,)]
        @test !Ksum.domain([0.6]) && Ksum.domain([0.4])
        @test_throws ArgumentError reflect(K, [1, 2])
        @test_throws ArgumentError marginal(K, [1, 1])
        @test_throws ArgumentError K + gamma_cgf(1.0)
    end

    @testset "saddlepoint" begin
        x = [-0.5, -0.5]
        tau, info = saddlepoint(K, x)
        tau2, _ = spa_saddle(Kf, x)
        @test norm(tau - tau2) <= 1e-13 && info.decrement <= 1e-10
        s, _ = saddlepoint(gamma_cgf(3.0), 6.0)
        @test s ≈ 1 - 3/6
        @test_throws ArgumentError saddlepoint(K, 1.0)
    end

    @testset "bivariate tails and densities (Wang's example)" begin
        x = [-0.5, -0.5]
        r = tailprob(K, x; order=5, lower=true)
        Kneg = (nu, t) -> (-1)^sum(nu)*Kf(nu, -t)
        r2 = spa_orthant2(Kneg, -x, 5)
        @test reldiff(r.F, r2.Fup) <= 1e-12 && r.status == "ok"
        @test norm(r.tau - spa_saddle(Kf, x)[1]) <= 1e-13
        f = density(K, x; order=4); f2 = spa_bi(Kf, x, 4)
        @test reldiff(f.f, f2.pdf) <= 1e-12 && f.status == "ok"
        # upper tail at a point where the saddlepoint has a negative component (inclusion-exclusion)
        x = [-0.8, 0.5]
        r = tailprob(K, x; order=3); r2 = spa_orthant2(Kf, x, 3)
        @test any(r.tau .< 0) && reldiff(r.F, r2.Fup) <= 1e-12
        # mixed corner of a bivariate normal against the exact orthant probability
        S = [1.0 0.6; 0.6 2.0]
        r = tailprob(normal_cgf([0.0, 0.0], S), [0.5, -0.3]; order=2, signs=[1, -1])
        @test abs(r.F[1]/orthant2([0.5, 0.3], [1.0 -0.6; -0.6 2.0]) - 1) <= 1e-9 && r.F[2] ≈ r.F[1]
        @test_throws ArgumentError tailprob(K, [0.1, 0.2, 0.3])
        @test_throws ArgumentError tailprob(K, [0.1, 0.2]; order=0)
        @test_throws ArgumentError tailprob(K, [0.1, 0.2]; signs=[1, 0])
    end

    @testset "pairs of quadratic forms against exact values" begin
        mm = [15, 20, 10]; ll = [5, 10, 8]
        Kc = common_component_cgf(mm, ll)
        x = [55.0, 70.0]
        ex = bqfcc_exact(x[1], x[2], mm, ll)
        r = tailprob(Kc, x; order=4); f = density(Kc, x; order=4)
        ec = abs.((r.F .- ex.Fup) ./ ex.Fup); ep = abs.((f.f .- ex.pdf) ./ ex.pdf)
        @test ec[1] < 2e-2 && ec[2] < 3e-4 && ec[3] < 2e-5 && ec[4] < 3e-6
        @test ep[1] < 2e-3 && ep[2] < 6e-4 && ep[3] < 3e-5 && ep[4] < 2e-5
        rb = setprecision(BigFloat, 140) do
            tailprob(Kc, big.(x); order=4)
        end
        @test eltype(rb.F) === BigFloat && reldiff(Float64.(rb.F), r.F) <= 1e-11
        # lower corner, directly and through the derivative interface
        Kf2 = (nu, t) -> bqfcc_Kfun(nu, t, mm, ll)
        rl = tailprob(Kc, [40.0, 60.0]; order=3, lower=true)
        Kn2 = (nu, t) -> (-1)^sum(nu)*Kf2(nu, -t)
        @test reldiff(rl.F, spa_orthant2(Kn2, [-40.0, -60.0], 3).Fup) <= 1e-12
        @test abs(rl.F[3] - bqfcc_exact(40.0, 60.0, mm, ll).Flo)/rl.F[3] < 1e-4
    end

    @testset "three and four dimensions" begin
        m3 = [15, 20, 25, 10]; l3 = [5, 10, 0, 8]
        K3 = common_component_cgf(m3, l3)
        x3 = [80.0, 95.0, 100.0]; ex3 = mvcc_exact(x3, m3, l3)
        r3 = tailprob(K3, x3; order=4)
        e3 = abs.((r3.F .- ex3.Fup) ./ ex3.Fup)
        @test e3[1] < 2e-2 && e3[2] < 3e-4 && e3[3] < 2e-5 && e3[4] < 2e-6
        @test r3.info.err[4] <= 1e-6*r3.F[4]
        f3 = density(K3, x3; order=4)
        ef = abs.((f3.f .- ex3.pdf) ./ ex3.pdf)
        @test ef[1] < 3e-2 && ef[2] < 2e-3 && ef[3] < 5e-5 && ef[4] < 3e-5 && f3.status == "ok"
        x3b = [55.0, 70.0, 75.0]; ex3b = mvcc_exact(x3b, m3, l3)
        r3b = tailprob(K3, x3b; order=3)                       # inclusion-exclusion
        @test any(r3b.tau .< 0)
        @test abs(r3b.F[3] - ex3b.Fup)/ex3b.Fup < 3e-5
        # Wishart diagonal through the generic log-determinant against the trace recursion
        V = 0.75 .* Matrix(1.0I, 3, 3) .+ 0.25
        Kw = wishart_diag_cgf(V, 5); Kwf = wishdiag_kfun(V, 5)
        t = [0.05, -0.02, 0.03]
        D = derivatives(Kw, t, 6)
        @test all(abs(D[nu] - Kwf(nu, t)) <= 1e-13*max(1, abs(D[nu])) for nu in keys(D))
        @test derivatives(Kw, [0.5, 0.5, 0.5], 2) === nothing
        rw = tailprob(Kw, [6.5, 6.5, 7.5]; order=3); rwf = spa_tri(Kwf, [6.5, 6.5, 7.5], 3)
        @test reldiff(rw.F, rwf.Fup) <= 1e-11
        # quadratic forms with a mean vector: sum of noncentral chi-squared parts
        Kq = quadratic_forms_cgf([Matrix(1.0I, 3, 3)[:, 1:2]*Matrix(1.0I, 3, 3)[1:2, :],
                                  Matrix(1.0I, 3, 3)[:, 3:3]*Matrix(1.0I, 3, 3)[3:3, :]]; mu=[1.0, 2.0, 0.5])
        Dq = derivatives(Kq, [0.1, -0.2], 4)
        Kx = CGF(t -> ncx2_cgf(2, 5.0)([t[1]]) + ncx2_cgf(1, 0.25)([t[2]]), 2)
        Dx = derivatives(Kx, [0.1, -0.2], 4)
        @test all(abs(Dq[nu] - Dx[nu]) <= 1e-12*max(1, abs(Dx[nu])) for nu in keys(Dq))
        # four dimensions: density exact in the sense of the expansion, orthant probability by QMC
        m4 = [15, 20, 25, 30, 10]; l4 = [5, 10, 0, 3, 8]
        K4 = common_component_cgf(m4, l4); x4 = [70.0, 85.0, 95.0, 105.0]
        ex4 = mvcc_exact(x4, m4, l4)
        f4 = density(K4, x4; order=3)
        f4f = spa_mv((nu, t) -> mvcc_Kfun(nu, t, m4, l4), x4, 3; method=:none)
        @test reldiff(f4.f, f4f.pdf) <= 1e-11
        r4 = tailprob(K4, x4; order=3)
        @test abs(r4.F[3] - ex4.Fup) <= max(1e-4*ex4.Fup, 4r4.info.err[3])
    end

    @testset "univariate and ratios" begin
        Kg = gamma_cgf(3.0)
        ex = gamma_exact(6.0, 3)
        r = tailprob(Kg, 6.0; order=4); f = density(Kg, 6.0; order=4)
        @test abs(r.F[4] - ex.upp)/ex.upp < 1e-4 && abs(r.F[1] - ex.upp)/ex.upp < 2e-2
        @test abs(f.f[4] - ex.pdf)/ex.pdf < 1e-5 && abs(f.f[1] - ex.pdf)/ex.pdf < 3e-2
        rl = tailprob(Kg, 6.0; order=4, lower=true)
        @test abs(rl.F[4] - ex.cdf) < 1e-4*ex.upp && rl.F[4] ≈ 1 - r.F[4]
        Kx = ncx2_cgf(5, 10)
        @test reldiff(tailprob(Kx, 30.0; order=6).F, ncx2_spa(30.0, 5, 10, 6).upp) <= 1e-12
        @test reldiff(density(Kx, 30.0; order=6).f, ncx2_spa(30.0, 5, 10, 6).pdf) <= 1e-12
        # the noncentral F as a ratio of independent chi-squared variables
        m_, n_, lam = 4, 12, 3; fq = 2.5; c = fq*m_/n_
        KF = CGF(t -> ncx2_cgf(m_, lam)([t[1]]) + ncx2_cgf(n_)([t[2]]), 2;
                 domain=t -> t[1] < 0.5 && t[2] < 0.5)
        rf = ncf_spa(fq, m_, n_, lam, 4)
        rd = ratio_density(KF, c; order=4)
        @test reldiff(rd.f .* (m_/n_), rf.pdf) <= 1e-12
        rt = ratio_tailprob(KF, c; order=4, lower=true)
        @test reldiff(rt.F, rf.cdf) <= 1e-12
        ru = ratio_tailprob(KF, c; order=4)
        @test reldiff(ru.F, rf.upp) <= 1e-12
        @test_throws ArgumentError ratio_density(Kg, 1.0)
    end
end
