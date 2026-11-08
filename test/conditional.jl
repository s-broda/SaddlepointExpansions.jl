# Conditional expansions (Broda and Kan, "Conditional saddlepoint expansions of arbitrary order"):
# the printed rows of the tables of the paper (conditional_rows.txt, shared with the Python
# package) recomputed from the cumulant generating function and exact values, consistency checks
# (B empty, Gaussian vectors, the Dirichlet distribution, the derivative interface, the leading
# term), and the BigFloat path.
#
# The relative errors are printed with two significant digits; a computed value reproduces a
# printed one when it rounds to it, and values below 1e-10, at the rounding level of the exact
# values, are compared to 2e-12 absolute. The cells listed in `fix1`, `fix2` and `fix3` are not
# reproduced: the value printed in the paper carries a rounding or truncation error of the MATLAB
# program that produced the table, and the value given here is that of the expansion, the same in
# double precision and in BigFloat (256 bits). In the tables of the gamma sum and of the quadratic
# forms these are the polynomial
# substitution into the canonical coordinates (rounding, d = 3 and N = 4, and N = 4, q = 0.99 of
# the first block), and the truncation of the orthant integrals at u v + v^2/2 = 40, which leaves
# out the growth of the Hermite polynomials of high degree (order 5 at the medians, where u ≈ 0);
# in the table of the sums of ratios, Skovgaard's formula at the median, where 1/w - 1/z cancels.

const CROWS = let rows = Dict{Int,Vector{Tuple{Int,Vector{String}}}}()
    for ln in eachline(joinpath(@__DIR__, "conditional_rows.txt"))
        startswith(ln, "#") && continue
        f = split(ln)
        push!(get!(rows, parse(Int, f[1]), Tuple{Int,Vector{String}}[]), (parse(Int, f[2]), String.(f[3:end])))
    end
    rows
end
pval(s) = s == "-" ? NaN : parse(Float64, s)
function printed(c, p)
    abs(p) < 1e-10 && return abs(c - p) <= 2e-12
    e = floor(Int, log10(abs(p)) + 1e-9)
    return abs(c - p) <= 0.05*10.0^e*(1 + 1e-9)
end
# the cells of one row: true where reproduced (fixes: column => value of the expansion)
function rowcheck(comp, cells, fixes)
    ok = trues(length(comp))
    for k in eachindex(comp)
        p = haskey(fixes, k) ? fixes[k] : pval(cells[k])
        isnan(p) && continue
        ok[k] = printed(comp[k], p)
        haskey(fixes, k) && (ok[k] &= !printed(comp[k], pval(cells[k])))
    end
    return ok
end
lgam(x) = SaddlepointExpansions.loggamma(x)
betapdf(w, p, q) = exp((p - 1)*log(w) + (q - 1)*log1p(-w) + lgam(p + q) - lgam(p) - lgam(q))
const beta_inc = SaddlepointExpansions.beta_inc
const beta_inc_inv = SaddlepointExpansions.beta_inc_inv

@testset "Conditional expansions" begin
    @testset "gamma sum given its total (cond_beta_rows.tex)" begin
        fix1 = Dict(3 => Dict(5 => -6.7e-16), 5 => Dict(5 => -7.5e-7, 10 => 1.4e-6), 17 => Dict(4 => 8.3e-7),
                    18 => Dict(5 => 5.1e-6), 23 => Dict(5 => 4.4e-9), 34 => Dict(5 => -5.6e-5, 10 => 2.1e-5),
                    35 => Dict(4 => 5.3e-5, 5 => -7.2e-5, 9 => 5.0e-5, 10 => 2.1e-5), 40 => Dict(5 => -2.6e-9))
        nok = 0; ncell = 0
        for (i, (block, cells)) in enumerate(CROWS[1])
            N = parse(Int, cells[1]); q = parse(Float64, cells[2])
            if block <= 2
                a, b = block == 1 ? (1.0, 1.0) : (0.5, 3.0)
                K = CGF(t -> -N*a*log(1 - t[1] - t[2]) - N*b*log(1 - t[2]), 2; domain=t -> t[1] + t[2] < 1 && t[2] < 1)
                s2 = N*(a + b); w = beta_inc_inv(N*a, N*b, q)[1]; x = [w*s2, s2]
                Pl, Pu = beta_inc(N*a, N*b, w); fex = betapdf(w, N*a, N*b)/s2
            else
                a, b, c = 1.0, 2.0, 0.5
                K = CGF(t -> -N*a*log(1 - t[1] - t[2]) - N*b*log(1 - t[2] - t[3]) - N*c*log(1 - t[2]), 3;
                        domain=t -> t[1] + t[2] < 1 && t[2] + t[3] < 1 && t[2] < 1)
                s2 = N*(a + b + c); s3 = N*b; w = beta_inc_inv(N*a, N*c, q)[1]; x = [w*(s2 - s3), s2, s3]
                Pl, Pu = beta_inc(N*a, N*c, w); fex = betapdf(w, N*a, N*c)/(s2 - s3)
            end
            r = conditional_tailprob(K, x, 1; order=5)
            f = conditional_density(K, x, 1; order=5)
            @test r.status == "ok" && f.status == "ok"
            @test printed(Pu, pval(cells[3])) && round(r.tau[1]; digits=2) == pval(cells[4]) + 0.0
            ok = rowcheck(vcat((r.F .- Pu) ./ min(Pu, Pl), f.f ./ fex .- 1), cells[5:14], get(fix1, i, Dict()))
            nok += count(ok); ncell += length(ok)
            @test all(ok)
        end
        @test (nok, ncell) == (450, 450)
        # the symmetric case a = b: the expansions at q and 1 - q add up to one, at every order
        K = CGF(t -> -4log(1 - t[1] - t[2]) - 4log(1 - t[2]), 2; domain=t -> t[1] + t[2] < 1 && t[2] < 1)
        w = beta_inc_inv(4.0, 4.0, 0.01)[1]
        F1 = conditional_tailprob(K, [w*8, 8.0], 1; order=5).F
        F2 = conditional_tailprob(K, [(1 - w)*8, 8.0], 1; order=5).F
        @test maximum(abs.(F1 .+ F2 .- 1)) < 1e-14
    end

    @testset "a quadratic form given another (cond_bqf_rows.tex)" begin
        fix2 = Dict(3 => Dict(5 => 3.4e-8))
        function illinois(f, a, b)
            fa, fb = f(a), f(b); side = 0
            for _ in 1:200
                c = (a*fb - b*fa)/(fb - fa); fc = f(c)
                (abs(fc) < 1e-15 || abs(b - a) < 1e-13*abs(c)) && return c
                if sign(fc) == sign(fb)
                    b, fb = c, fc; side == 1 && (fa /= 2); side = 1
                else
                    a, fa = c, fc; side == -1 && (fb /= 2); side = -1
                end
            end
            return (a + b)/2
        end
        cases = (([15, 20, 10], [5, 10, 8]), ([40, 60, 30], [20, 10, 30]))
        nok = 0; ncell = 0
        for (i, (block, cells)) in enumerate(CROWS[2])
            m, lam = cases[block]
            c2 = parse(Float64, cells[1]); q = parse(Float64, cells[2])
            mu2 = m[2] + lam[2] + m[3] + lam[3]; sd2 = sqrt(2(m[2] + 2lam[2]) + 2(m[3] + 2lam[3]))
            x2 = mu2 + c2*sd2
            f0(u) = ncx2_exact(u, m[3], lam[3]).pdf; f2(u) = ncx2_exact(u, m[2], lam[2]).pdf
            den = quadgk(u -> f0(u)*f2(x2 - u), 0, x2; rtol=1e-13)[1]
            br(x1) = x1 < x2 ? (0.0, x1, x2) : (0.0, x2)
            ex(x1, g) = quadgk(u -> f0(u)*f2(x2 - u)*g(ncx2_exact(x1 - u, m[1], lam[1])), br(x1)...; rtol=1e-13)[1]/den
            x1 = illinois(x -> ex(x, e -> e.cdf) - q, 1e-6, mu2 + m[1] + lam[1] + 20sd2)
            Pu, Pl = ex(x1, e -> e.upp), ex(x1, e -> e.cdf)
            fex = quadgk(u -> f0(u)*f2(x2 - u)*ncx2_exact(x1 - u, m[1], lam[1]).pdf, 0, min(x1, x2); rtol=1e-13)[1]/den
            K = common_component_cgf(m, lam)
            r = conditional_tailprob(K, [x1, x2], 1; order=5)
            f = conditional_density(K, [x1, x2], 1; order=5)
            @test printed(Pu, pval(cells[3])) && round(r.tau[1]; digits=2) == pval(cells[4]) + 0.0
            ok = rowcheck(vcat((r.F .- Pu) ./ min(Pu, Pl), f.f ./ fex .- 1), cells[5:14], get(fix2, i, Dict()))
            nok += count(ok); ncell += length(ok)
            @test all(ok)
        end
        @test (nok, ncell) == (300, 300)
    end

    @testset "the mean of N independent ratios of quadratic forms (sph_rows_*.tex)" begin
        # P(Rbar >= x/N) = P(X >= x | D_1 = ... = D_N = 1), X = sum_i v_i'A_iv_i, D_i = v_i'v_i
        function lams(N, T0)
            out = Vector{Vector{Float64}}()
            for T in T0+1:T0+N
                A = zeros(T, T)
                for t in 1:T-1
                    A[t, t+1] = A[t+1, t] = 0.5
                end
                Q = nullspace(ones(1, T))
                push!(out, eigvals(Symmetric(Q'*A*Q)))
            end
            return out
        end
        # the exact upper quantiles of S = N Rbar at p = 0.5, 0.1, 0.05, 0.01, 0.001, by the
        # Gil-Pelaez inversion of the product of the characteristic functions of the ratios
        # (the program sph_exact.m of the paper, ported below and checked on the second block)
        qex = Dict(1 => (-0.649984406337777, 0.173640308684513, 0.404657822568259, 0.830136764358454, 1.28888641542161),
                   2 => (-0.28101745873196, 0.342338885972127, 0.518020283322166, 0.844515706483764, 1.20359268966255),
                   3 => (-0.179727297726595, 0.335098902241048, 0.480370305892609, 0.751043128437577, 1.05038179960608),
                   4 => (-1.17188025603077, -0.0622176078581697, 0.251186562311314, 0.834501869250704, 1.47747984786112),
                   5 => (-0.525777371167559, 0.329433323857853, 0.571415079075747, 1.02356949383106, 1.52620380057527))
        design = Dict(1 => (5, 5), 2 => (5, 15), 3 => (5, 25), 4 => (10, 4), 5 => (10, 14))
        ps = (0.5, 0.1, 0.05, 0.01, 0.001)
        # Skovgaard's formula at the median: its value in BigFloat
        fix3 = Dict((3, 1) => Dict(2 => -1.4e-4))
        Phi(x) = SaddlepointExpansions.normcdf(x); phi(x) = SaddlepointExpansions.normpdf(x)
        nok = 0; ncell = 0
        rowsof = Dict{Int,Vector{Vector{String}}}()
        for (block, cells) in CROWS[3]
            push!(get!(rowsof, block, Vector{String}[]), cells)
        end
        for block in 1:5
            N, T0 = design[block]
            lam = lams(N, T0); d = N + 1
            K = CGF(t -> -sum(sum(log(1 - 2*(t[1]*l + t[i+1])) for l in lam[i]) for i in 1:N)/2, d;
                    domain=t -> all(all(1 - 2*(t[1]*l + t[i+1]) > 0 for l in lam[i]) for i in 1:N))
            mu = sum(sum(a)/length(a) for a in lam)
            va = sum((sum(a)^2 + 2sum(abs2, a))/(length(a)*(length(a) + 2)) - (sum(a)/length(a))^2 for a in lam)
            for (k, p) in enumerate(ps)
                cells = rowsof[block][k]
                x = qex[block][k]; xv = vcat(x, ones(N))
                r = conditional_tailprob(K, xv, 1; order=(N == 5 ? 4 : 2))
                @test r.status == "ok"
                @test round(x/N; digits=4) == pval(cells[2]) + 0.0 && round(r.tau[1]; digits=3) == pval(cells[4]) + 0.0
                # the normal approximation, Skovgaard's approximation and the leading term
                function lrhr(xv, prec)
                    setprecision(BigFloat, prec) do
                        xx = prec > 53 ? big.(xv) : xv
                        rr = prec > 53 ? conditional_tailprob(K, xx, 1; order=1) : r
                        tau, taut = rr.tau, rr.taut
                        D = derivatives(K, tau, 2); Dt = derivatives(K, taut, 2)
                        e(i) = ntuple(j -> j == i ? 1 : 0, d)
                        K2h = [D[e(i) .+ e(j)] for i in 1:d, j in 1:d]; K2t = [Dt[e(i+1) .+ e(j+1)] for i in 1:N, j in 1:N]
                        w = sign(tau[1])*sqrt(2*((Dt[ntuple(_ -> 0, d)] - dot(taut, xx)) - (D[ntuple(_ -> 0, d)] - dot(tau, xx))))
                        z = tau[1]*sqrt(det(K2h)/det(K2t))
                        u = tau[1]*sqrt(K2h[1, 1] - dot(K2h[1, 2:end], K2h[2:end, 2:end] \ K2h[2:end, 1]))
                        Float64(Phi(-w) - phi(w)*(1/w - 1/z)), Float64(u/z*exp((u^2 - w^2)/2)*Phi(-u))
                    end
                end
                Plr, Phr = lrhr(xv, 53)
                p == 0.5 && (Plr = lrhr(xv, 256)[1])
                p < 0.5 && @test abs(r.info.Fp[1]/Phr - 1) < 1e-12       # (eq:robcond) of the paper
                Pn = Phi(-(x - mu)/sqrt(va))
                comp = vcat([Pn, Plr, Phr] .- p, r.F .- p) ./ p
                ok = rowcheck(comp, cells[5:4+length(comp)], get(fix3, (block, k), Dict()))
                nok += count(ok); ncell += length(ok)
                @test all(ok)
            end
        end
        @test (nok, ncell) == (155, 155)
        # the exact quantiles of the second block, recomputed
        let lam = lams(5, 15), (xw, ww) = SaddlepointExpansions.gauleg(12000), (xg, wg) = SaddlepointExpansions.gauleg(40)
            om = 150 .* (xw .+ 1); wom = 150 .* ww
            Phic = ones(ComplexF64, length(om))
            for a in lam
                a = sort(a); nodes = Float64[]; wts = Float64[]
                for k in 1:length(a)-1
                    ns = max(1, ceil(Int, (a[k+1] - a[k])/0.1)); e = range(a[k], a[k+1]; length=ns+1)
                    for q in 1:ns
                        append!(nodes, (e[q] + e[q+1])/2 .+ (e[q+1] - e[q])/2 .* xg); append!(wts, (e[q+1] - e[q])/2 .* wg)
                    end
                end
                fr = [quadgk(w -> real(sum(1 ./ (1 .- 2im*w .* (a .- r)))*prod((1 .- 2im*w .* (a .- r)).^(-0.5))), 0, Inf;
                             rtol=1e-11, atol=1e-14)[1]/pi for r in nodes]
                @test abs(dot(wts, fr) - 1) < 1e-10
                Phic .*= exp.(im .* om .* nodes') * (wts .* fr)
            end
            upper(x) = 1/2 + sum(wom .* imag.(exp.(-im .* om .* x) .* Phic) ./ om)/pi
            @test maximum(abs.([upper(qex[2][k]) for k in 1:5] ./ collect(ps) .- 1)) < 1e-9
        end
    end

    @testset "logistic regression, lattice (logit_rows.tex)" begin
        n = 30; X = [ones(Int, n) mod.(0:n-1, 5) mod.(0:n-1, 3)]
        mx = vec(sum(X; dims=1))
        pm = zeros(mx[1] + 1, mx[2] + 1, mx[3] + 1); pm[1, 1, 1] = 1.0
        for j in 1:n                                     # the exact joint pmf of S under beta = 0
            pm .= (pm .+ circshift(pm, Tuple(X[j, :]))) ./ 2
        end
        K = logistic_regression_cgf(X)
        Kf = SaddlepointExpansions._kfun(K, 9); Kw = SaddlepointExpansions._kwhite(K, 9)
        nok = 0; ncell = 0; i = 0
        for (k0, k2) in ((15, 15), (10, 12), (12, 9))
            q = pm[k0+1, :, k2+1]; q ./= sum(q)
            Qu = reverse(cumsum(reverse(q))); Ql = cumsum(q)
            for k1 in unique([findfirst(>=(tg), Ql) - 1 for tg in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)])
                i += 1; cells = CROWS[4][i][2]
                Pu, Pl, pk = Qu[k1+1], Ql[k1+1], q[k1+1]
                x = Float64[k0, k1, k2]
                r = conditional_tailprob(K, x, 2; order=4, lattice=true)
                f = conditional_density(K, x, 2; order=4, lattice=true)
                # the last column: the MATLAB program keeps the expansion of s/(2 sinh(s/2)) about
                # the origin to s^(2 order - 2) = s^6 at every order, so its "second-order" value has
                # the polynomial of degree 6; Kolassa's truncation (sinh = :origin) keeps s^2
                r6 = SaddlepointExpansions._spa_cond(Kf, Kw, x, [2], 4; density=false, lattice=true, sinh=:origin, sinhdeg=6)
                ro = conditional_tailprob(K, x, 2; order=4, lattice=true, sinh=:origin)
                @test parse(Int, cells[1]) == k1 && printed(Pu, pval(cells[2])) && round(r.tau[2]; digits=2) == pval(cells[3]) + 0.0
                @test r.status == "ok"
                ok = rowcheck(vcat((r.F .- Pu) ./ min(Pu, Pl), f.f ./ pk .- 1, (r6.P[2] - Pu)/min(Pu, Pl)), cells[4:12], Dict())
                nok += count(ok); ncell += length(ok)
                @test all(ok)
                @test abs(ro.F[2] - Pu)/min(Pu, Pl) < 4e-3 && abs(ro.F[2]/r.F[2] - 1) < 1e-3
                # the lower tail P(S_1 <= k_1 - 1) is the complement, at every order
                rl = conditional_tailprob(K, x .- [0, 1, 0], 2; order=4, lattice=true, lower=true)
                @test maximum(abs.(rl.F .+ r.F .- 1)) < 1e-14
            end
        end
        @test (nok, ncell) == (189, 189)
    end

    @testset "consistency" begin
        rd(a, b) = maximum(abs.(a .- b) ./ abs.(b))
        # B empty: the expansions of the package for the orthant probability and the density
        Kg = gamma_cgf(3.0)
        for xx in (6.0, 1.5)
            @test rd(conditional_tailprob(Kg, [xx], 1; order=4).F, tailprob(Kg, xx; order=4).F) < 1e-13
            @test rd(conditional_density(Kg, [xx], 1; order=4).f, density(Kg, xx; order=4).f) < 1e-14
        end
        K5 = exponential_sums_cgf([1 0; 1 1; 0 1]; n=5)
        for xx in ([3.0, 3.0], [2.5, 3.0], [1.2, 3.5], [0.8, 0.9])     # tau >= 0, tau_1 = 0, mixed, negative
            @test rd(conditional_tailprob(K5, xx, [1, 2]; order=4).F, tailprob(K5, xx; order=4).F) < 1e-12
            @test rd(conditional_density(K5, xx, 1:2; order=4).f, density(K5, xx; order=4).f) < 1e-14
        end
        # Gaussian vectors: exact at every order, one, two and three tail components, every sign of
        # the saddlepoint (the inclusion-exclusion marginalizes the components of N \ T)
        Sig = [2.0 0.6 0.5 0.3; 0.6 1.5 -0.4 0.2; 0.5 -0.4 1.0 0.1; 0.3 0.2 0.1 0.8]; mu = [0.5, -1.0, 0.2, 0.3]
        Kn = normal_cgf(mu, Sig)
        function condnorm(x, A)
            B = setdiff(1:4, A); C = Sig[A, B]/Sig[B, B]
            V = Sig[A, A] - C*Sig[B, A]
            return mu[A] + C*(x[B] - mu[B]), (V + V')/2
        end
        for (x, A) in (([1.0, 0.0, 0.5, 0.0], [1]), ([-1.0, 0.0, 0.5, 0.0], [1]), ([1.5, 0.0, 0.5, 0.3], [1, 2]),
                       ([0.2, -2.5, 0.5, 0.3], [1, 2]), ([-1.0, -2.5, 0.5, 0.3], [2, 1]), ([1.2, -0.8, 0.9, 0.3], [1, 2, 3]))
            Av = sort(A); m, V = condnorm(x, Av)
            ex = length(A) == 1 ? SaddlepointExpansions.normcdf(-(x[A[1]] - m[1])/sqrt(V[1, 1])) :
                 length(A) == 2 ? orthant2(x[Av] .- m, V) : orthant3(x[Av] .- m, V)
            r = conditional_tailprob(Kn, x, A; order=(length(A) == 3 ? 2 : 3))
            @test maximum(abs.(r.F ./ ex .- 1)) < 1e-13 && r.status == "ok"
            fex = exp(-dot(x[Av] - m, V \ (x[Av] - m))/2)/sqrt((2pi)^length(A)*det(V))
            @test maximum(abs.(conditional_density(Kn, x, A; order=3).f ./ fex .- 1)) < 1e-13
        end
        @test any(conditional_tailprob(Kn, [0.2, -2.5, 0.5, 0.3], [1, 2]).tau[1:2] .< 0)
        # lower tails and mixed corners: P(X_1 <= x_1, X_2 >= x_2 | X_B) by the reflection
        x = [0.2, -2.5, 0.5, 0.3]; m, V = condnorm(x, [1, 2])
        rm = conditional_tailprob(Kn, x, [1, 2]; order=2, signs=[-1, 1])
        @test abs(rm.F[2]/orthant2([-(x[1] - m[1]), x[2] - m[2]], [V[1, 1] -V[1, 2]; -V[1, 2] V[2, 2]]) - 1) < 1e-13
        rl = conditional_tailprob(Kn, x, [1, 2]; order=2, lower=true)
        @test abs(rl.F[2]/orthant2(m .- x[1:2], V) - 1) < 1e-13
        # two tail components given the total: the Dirichlet distribution, exact by quadrature
        al = (5.0, 7.5, 10.0)
        Kd = CGF(t -> -al[1]*log(1 - t[1] - t[3]) - al[2]*log(1 - t[2] - t[3]) - al[3]*log(1 - t[3]), 3;
                 domain=t -> t[1] + t[3] < 1 && t[2] + t[3] < 1 && t[3] < 1)
        s = sum(al); lgD = lgam(s) - sum(lgam, al)
        for (w1, w2) in ((0.35, 0.45), (0.15, 0.40), (0.30, 0.25), (0.15, 0.25))
            ex = quadgk(v -> betapdf(v, al[1], al[2] + al[3])*beta_inc(al[2], al[3], w2/(1 - v))[2], w1, 1 - w2; rtol=1e-13)[1]
            fex = exp(lgD + (al[1] - 1)*log(w1) + (al[2] - 1)*log(w2) + (al[3] - 1)*log(1 - w1 - w2))/s^2
            r = conditional_tailprob(Kd, [w1*s, w2*s, s], [1, 2]; order=4)
            e = abs.(r.F ./ ex .- 1)
            @test e[1] < 2e-2 && e[2] < 1e-3 && e[3] < 2e-5 && e[4] < 1e-5 && e[4] < e[1]
            @test r.status == "ok" && maximum(r.info.kappa) < 1e3 && r.info.err[end] < 1e-9*r.F[end]
            fd = conditional_density(Kd, [w1*s, w2*s, s], [1, 2]; order=4).f ./ fex .- 1
            @test abs(fd[1] - 3.3e-2) < 1e-3 && abs(fd[4]) < 2e-6           # the same at every point
        end
        # the derivative interface: closed-form derivatives with the polynomial substitution into the
        # canonical coordinates, and with Kwhite; the leading term (eq:robcond)
        N = 16; a, b, c = 1.0, 2.0, 0.5
        function Kf3(nu, t)
            n = sum(nu); p1 = 1 - t[1] - t[2]; p2 = 1 - t[2] - t[3]; p3 = 1 - t[2]
            (p1 <= 0 || p2 <= 0 || p3 <= 0) && return NaN
            n == 0 && return -N*(a*log(p1) + b*log(p2) + c*log(p3))
            v = 0.0
            nu[3] == 0 && (v += a*factorial(n - 1)/p1^n)
            nu[1] == 0 && (v += b*factorial(n - 1)/p2^n)
            (nu[1] == 0 && nu[3] == 0) && (v += c*factorial(n - 1)/p3^n)
            return N*v
        end
        K3 = CGF(t -> -N*a*log(1 - t[1] - t[2]) - N*b*log(1 - t[2] - t[3]) - N*c*log(1 - t[2]), 3)
        x3 = [19.0, 14*N/4, 8*N/4]                           # above the conditional mean 16
        r1 = spa_cond(Kf3, x3, 1, 4)
        r2 = conditional_tailprob(K3, x3, 1; order=4); f2 = conditional_density(K3, x3, 1; order=4)
        r3 = spa_cond(SaddlepointExpansions._kfun(K3, 9), x3, 1, 4; Kwhite=SaddlepointExpansions._kwhite(K3, 9))
        @test rd(r1.P, r2.F) < 1e-10 && rd(r1.f, f2.f) < 1e-10
        @test r3.P == r2.F && r3.f == f2.f && r3.Pp == r2.info.Fp
        @test r1.Pp[2:2:end] == r1.P && length(r1.Pp) == 8
        @test r1.tau ≈ r2.tau && r1.taut ≈ r2.taut && r1.taut[1] == 0
        D = derivatives(K3, r2.tau, 2); Dt = derivatives(K3, r2.taut, 2); e(i) = ntuple(j -> j == i ? 1 : 0, 3)
        H = [D[e(i) .+ e(j)] for i in 1:3, j in 1:3]; Ht = H[2:3, 2:3]
        Ht = [Dt[e(i) .+ e(j)] for i in 2:3, j in 2:3]
        w = sqrt(2*((Dt[(0, 0, 0)] - dot(r2.taut, x3)) - (D[(0, 0, 0)] - dot(r2.tau, x3))))
        u = r2.tau[1]*sqrt(H[1, 1] - dot(H[1, 2:3], H[2:3, 2:3] \ H[2:3, 1])); z = r2.tau[1]*sqrt(det(H)/det(Ht))
        @test abs(r2.info.Fp[1]/(u/z*exp((u^2 - w^2)/2)*SaddlepointExpansions.normcdf(-u)) - 1) < 1e-13
        @test r2.tau[1] > 0 && r2.info.err[end] < 1e-12*r2.F[end] && r2.info.nG == 64
        # arguments
        @test_throws ArgumentError conditional_tailprob(K3, x3, 4)
        @test_throws ArgumentError conditional_tailprob(K3, x3, [1, 1])
        @test_throws ArgumentError conditional_tailprob(K3, x3, 1; order=0)
        @test_throws ArgumentError conditional_tailprob(K3, x3, [1, 2]; signs=[1])
        @test_throws ArgumentError conditional_tailprob(K3, x3, 1; sinh=:other)
        @test_throws ArgumentError conditional_density(K3, x3[1:2], 1)
    end

    @testset "BigFloat" begin
        # the gamma sum, d = 3, N = 4, q = 0.99, where the double-precision program of the paper failed
        K = CGF(t -> -4log(1 - t[1] - t[2]) - 8log(1 - t[2] - t[3]) - 2log(1 - t[2]), 3;
                domain=t -> t[1] + t[2] < 1 && t[2] + t[3] < 1 && t[2] < 1)
        w = beta_inc_inv(4.0, 2.0, 0.99)[1]; x = [6w, 14.0, 8.0]
        Pl, Pu = beta_inc(4.0, 2.0, w)
        r = conditional_tailprob(K, x, 1; order=5)
        rb, fb = setprecision(BigFloat, 256) do
            conditional_tailprob(K, big.(x), 1; order=5), conditional_density(K, big.(x), 1; order=5)
        end
        @test eltype(rb.F) === BigFloat && eltype(fb.f) === BigFloat && precision(rb.F[1]) == 256
        @test maximum(abs.(Float64.(rb.F) .- r.F)) < 1e-12
        @test abs(Float64((rb.F[5] - Pu)/min(Pu, Pl)) + 7.1735e-5) < 1e-8
        @test abs(Float64(fb.f[5])/(betapdf(w, 4.0, 2.0)/6) - 1 - 2.0734e-5) < 1e-9
        @test all(iszero, rb.info.err)                   # one tail component: the exact recursion
        # two tail components: quadrature in the working precision
        Kd = CGF(t -> -5log(1 - t[1] - t[3]) - 7.5log(1 - t[2] - t[3]) - 10log(1 - t[3]), 3)
        xd = [0.3*22.5, 0.45*22.5, 22.5]
        r2 = conditional_tailprob(Kd, xd, [1, 2]; order=3)
        r2b = setprecision(BigFloat, 128) do
            conditional_tailprob(Kd, big.(xd), [1, 2]; order=3)
        end
        @test maximum(abs.(Float64.(r2b.F) ./ r2.F .- 1)) < 1e-13
        # lattice, BigFloat
        Kl = logistic_regression_cgf([ones(Int, 30) mod.(0:29, 5) mod.(0:29, 3)])
        rl = conditional_tailprob(Kl, [15.0, 37, 15], 2; order=3, lattice=true)
        rlb = setprecision(BigFloat, 128) do
            conditional_tailprob(Kl, big.([15.0, 37, 15]), 2; order=3, lattice=true)
        end
        @test maximum(abs.(Float64.(rlb.F) ./ rl.F .- 1)) < 1e-13
    end
end
