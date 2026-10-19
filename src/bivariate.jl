# Bivariate expansions: density (Theorem 3), orthant probability (Theorem 1), density of a pair
# of ratios (Corollary 1).

"""
    spa_bi(Kfun, x, order; tau0=nothing, gamfun=nothing, nG=200, Imethod=:auto, Kwhite=nothing,
           gamwhite=nothing, direct=false, logscale=0, qtol=1e-8, nGmax=4nG)
        -> (; pdf, Fup, tau, canc, kappa, qerr, status, nG, coef_err_pdf, coef_err_cdf, saddle)

Bivariate saddlepoint approximations of orders `1:order` at the point `x = [x1, x2]`.
`Kfun(nu, t)` returns the derivative of order `nu = (nu1, nu2)` (a tuple of integers) of the
cumulant generating function ``K`` at `t = [t1, t2]`; `nu = (0, 0)` gives ``K(t)``. It must
return `NaN` outside the domain of ``K``.

* `pdf[M]`: density expansion with the terms `m ≤ M-1` (relative error ``O(n^{-M})``);
* `Fup[M]`: ``P(X_1 ≥ x_1, X_2 ≥ x_2)`` with the terms `m ≤ 2M-1`, valid when the saddlepoint
  `tau` is in the closed positive orthant (`NaN` otherwise; use [`spa_orthant2`](@ref));
* `tau`: the saddlepoint, found by [`spa_saddle`](@ref) (Newton's method with backtracking,
  convergence measured by the Newton decrement; an error is raised if it does not converge);
* `canc[M]`: the largest |term| among the terms of `Fup[M]` divided by ``I(0)``;
* `kappa[M]`: (sum of |terms| of `Fup[M]`)/|sum|; about ``\\log_{10}κ`` digits are lost to
  cancellation in the final sum (it does not measure errors in the coefficients, see
  `coef_err_cdf`);
* `qerr[M]`: quadrature check of `Fup[M]`: its relative change with half the nodes, all retained
  terms included (`0` for the exact method); while `qerr[order] > qtol` the number of nodes of
  the Gauss-Legendre rule is doubled, up to `nGmax`. A comparison of two quadratures, not a bound.
  `Fup` and `qerr` are `NaN` when the leading integral ``I(0)`` cannot be resolved by the
  quadrature (it comes out zero or not finite), instead of a misleading zero;
* `status`: `"ok"` or one or more of (joined by `"; "`) `"negative saddlepoint component (use
  spa_orthant2)"`, `"leading integral not resolved"`, `"integration not resolved"`,
  `"coefficients inaccurate"` (a coefficient-error estimate above ``10^{-6}``), `"probability
  below the double-precision range (use logscale)"` (the Chernoff bound ``e^{K(τ)-τ'x-c}`` is below
  `floatmin`; `Fup` is then zero);
* `nG`: the number of Gauss-Legendre nodes used;
* `coef_err_pdf[M]`, `coef_err_cdf[M]` (`Float64` with the whitened evaluation; `NaN` otherwise):
  estimates of the relative errors of `pdf[M]` and `Fup[M]` caused by rounding in the formation of
  the whitened Taylor coefficients (first-order propagation of random perturbations of the size of
  the rounding errors of the substitution; with `Kwhite` only the rounding of the supplied
  derivatives);
* `saddle`: the diagnostics of [`spa_saddle`](@ref).

With `gamfun`, `pdf` is the density of a pair of ratios (Section 4): `gamfun(tau)` returns a
tuple `(G0, G1)` of the Taylor coefficients at `tau` of the amplitude functions ``g_0``, ``g_1``,
`Gk[mu1+1, mu2+1]` the coefficient of ``t^μ`` in ``g_k(τ+t)``, which enters ``b_j(g_k)`` of
Corollary 1, ``|μ| ≤ 2(\\mathrm{order}-1)``, and `x` should be `[0, 0]`.

The arithmetic is done in the floating-point type of `x` (and `tau0`); pass `BigFloat` values
(and a `Kfun` that is accurate in that precision) for variable precision. `Imethod` selects the
computation of the integrals ``I(ν)``: `:quad` (default for `Float64`; Gauss-Hermite or
Gauss-Legendre quadrature with `nG` nodes per coordinate, Appendix B) or `:exact` (default for
other types; binomial sum (14) and recursion of Lemma 2, which cancel in double precision).

Whitened evaluation (`:quad`). With ``K''(τ)=LL'`` (Cholesky) and ``T=L^{-T}``, the terms of each
order are evaluated for ``Y=L^{-1}X``: the Taylor coefficients of ``K_Y(u)=K(Tu)`` at
``\\hat u=L'τ`` replace those of ``K`` at ``τ``, the Hermite polynomials ``He_ν(w,K'')`` are
replaced by products of univariate ones in ``v=L^{-1}w``, and the orthant integrals ``I(ν)`` by
integrals of these products over the same orthant, at the same quadrature nodes. The sums of the
terms of a given order are the same in exact arithmetic; in the whitened form the terms cancel
much less when the components of ``X`` are strongly correlated under the tilted distribution
(Appendix B of the paper), provided the whitened coefficients are formed accurately. The method
`:exact` keeps the original coordinates.

Keywords:
* `Kwhite`: optional function `(nu, u, T) -> ` derivative of order `nu` at `u` of
  ``K_Y(u)=K(Tu)``, the cumulant generating function of ``Y=T'X``, for a ``2×2`` matrix `T` (with
  ``T=L^{-T}``, ``Y=L^{-1}X``); when absent, the Taylor coefficients of ``K_Y`` are obtained from
  those of ``K`` by polynomial substitution, which loses digits when ``K''`` is ill conditioned
  (`coef_err_pdf`, `coef_err_cdf` estimate the loss);
* `gamwhite`: optional function `(u, T) -> (G0, G1)`, the amplitude coefficients of `gamfun`
  expressed in the whitened variable (otherwise substitution);
* `direct = true` evaluates the orthant expansion also when `tau` has negative components, by the
  orthant-integral form of ``I(ν)`` (the analytic continuation of the formula; the "direct"
  evaluation of Section 3.3, which has the wrong exponent; use `spa_orthant2` for the probability
  itself);
* `logscale = c` returns `Fup` scaled by ``e^{-c}``, for probabilities beyond the range of double
  precision;
* `qtol`, `nGmax`: tolerance and limit of the node escalation (see `qerr`).

In the Gauss-Legendre branch the exponential factor ``E_n`` of (12) is combined with the normal
density at the nodes in the log domain, and the density prefactor is formed from the logarithms of
the diagonal of the Cholesky factor of ``K''``, so that the products are formed without overflow or
underflow of their factors. The nodes in ``z_1`` are split, with `nG` nodes on each piece, at the
edges of the boundary layer that the truncation ``z_2 ≥ t_2`` creates when the correlation is near
``±1`` and where the density of ``z_1`` decays quickly because ``t_1`` is far in its tail.
"""
function spa_bi(Kfun, x::AbstractVector, order::Integer; tau0=nothing, gamfun=nothing,
                nG::Integer=200, Imethod::Symbol=:auto, Kwhite=nothing, gamwhite=nothing,
                direct::Bool=false, logscale::Real=0, qtol::Real=1e-8, nGmax::Integer=4nG)
    length(x) == 2 && all(isfinite, x) || throw(ArgumentError("spa_bi: x must be a finite vector of length 2"))
    order >= 1 || throw(ArgumentError("spa_bi: order must be a positive integer"))
    nG >= 4 || throw(ArgumentError("spa_bi: nG must be an integer >= 4"))
    T = float(tau0 === nothing ? eltype(x) : promote_type(eltype(x), eltype(tau0)))
    x = T.(collect(x))
    t0 = tau0 === nothing ? zeros(T, 2) : T.(collect(tau0))
    tau, sinfo = spa_saddle(Kfun, x, t0)
    p = 2order - 1
    Kd = fill(T(NaN), p + 3, p + 3)
    for k in 0:p+2, n1 in 0:k
        Kd[n1+1, k-n1+1] = Kfun((n1, k - n1), tau)
    end
    G = gamfun === nothing ? nothing : gamfun(tau)
    r = spa_bi_at(Kd, tau, x, order; G, nG, Imethod, Kwhite, gamwhite, direct, logscale, qtol, nGmax)
    return (; pdf=r.pdf, Fup=r.Fup, tau, canc=r.canc, kappa=r.kappa, qerr=r.qerr, status=r.status,
            nG=r.nG, coef_err_pdf=r.coef_err_pdf, coef_err_cdf=r.coef_err_cdf, saddle=sinfo)
end

"""
    spa_bi_at(Kd, tau, x, order; G=nothing, nG=200, Imethod=:auto, Kwhite=nothing,
              gamwhite=nothing, direct=false, logscale=0, qtol=1e-8, nGmax=4nG)
        -> (; pdf, Fup, canc, kappa, qerr, status, nG, coef_err_pdf, coef_err_cdf)

The expansions of [`spa_bi`](@ref) from the derivatives at a given saddlepoint `tau`:
`Kd[n1+1, n2+1]` ``=K^{(n_1,n_2)}(τ)``, ``n_1+n_2 ≤ 2\\,\\mathrm{order}+1``, ``K'(τ)=x``.
`G = (G0, G1)` are the amplitude coefficients of a pair of ratios at `tau` (see `spa_bi`).
Useful when the saddlepoint and the derivatives have been computed separately, e.g. in high
precision. The other keywords are those of `spa_bi`.
"""
function spa_bi_at(Kd::AbstractMatrix, tau::AbstractVector, x::AbstractVector, order::Integer;
                   G=nothing, nG::Integer=200, Imethod::Symbol=:auto, Kwhite=nothing,
                   gamwhite=nothing, direct::Bool=false, logscale::Real=0, qtol::Real=1e-8,
                   nGmax::Integer=4nG)
    T = float(promote_type(eltype(Kd), eltype(tau), eltype(x)))
    Kd = T.(Kd); tau = T.(collect(tau)); x = T.(collect(x))
    meth = Imethod === :auto ? (T === Float64 ? :quad : :exact) : Imethod
    meth in (:quad, :exact) || throw(ArgumentError("spa_bi: Imethod must be :auto, :quad or :exact"))
    meth === :quad && T !== Float64 && throw(ArgumentError("spa_bi: Imethod = :quad needs Float64"))
    white = meth === :quad
    K0 = Kd[1, 1]
    K2 = [Kd[3, 1] Kd[2, 2]; Kd[2, 2] Kd[1, 3]]
    p = 2order - 1
    Nn = max(3p, 6(order - 1))                     # largest |nu| needed
    Lc = cholesky(Symmetric(K2)).L
    Tm = inv(Matrix(Lc'))                          # s = T u, T upper triangular
    uh = Lc'*tau                                   # saddlepoint of K_Y
    # theta[nu1+1,nu2+1] = Taylor coefficient of K (or of K_Y at uh), 3 <= |nu| <= p+2;
    # a[r][nu1+1,nu2+1] = [t^nu] G^r/r!
    theta = zeros(T, Nn + 1, Nn + 1)
    cerr = white && T === Float64                  # coefficient-error estimates (double precision)
    degm = [i + j for i in 0:Nn, j in 0:Nn]
    dth = zeros(T, Nn + 1, Nn + 1)
    if white && Kwhite !== nothing
        for k in 3:p+2, n1 in 0:k
            theta[n1+1, k-n1+1] = T(Kwhite((n1, k - n1), uh, Tm))/(factT(T, n1)*factT(T, k - n1))
        end
        cerr && (dth = 2eps(T) .* abs.(theta))
    else
        for k in 3:p+2, n1 in 0:k
            theta[n1+1, k-n1+1] = Kd[n1+1, k-n1+1]/(factT(T, n1)*factT(T, k - n1))
        end
        cerr && (dth = (degm .+ 2) .* eps(T) .* subs2(abs.(theta), abs.(Tm)))
        white && (theta = subs2(theta, Tm))
    end
    R = max(p, 2(order - 1))
    a = Vector{Matrix{T}}(undef, max(R, 1))
    a[1] = theta
    for r in 2:R
        a[r] = polyprod(theta, a[r-1], 3r, Nn) ./ r
    end
    # first-order propagation of random perturbations of the whitened coefficients of the size of
    # their rounding errors: da[r] = a[r-1] * dtheta, the derivative of G^r/r!
    ndraw = 2
    rng = _Lcg(1)
    da = [Vector{Matrix{T}}(undef, max(R, 1)) for _ in 1:ndraw]
    if cerr
        for q in 1:ndraw
            dt = dth .* [sign(_rand(rng) - 0.5) for _ in 1:Nn+1, _ in 1:Nn+1]
            da[q][1] = dt
            for r in 2:R
                da[q][r] = polyprod(dt, a[r-1], 3r, Nn)
            end
        end
    end
    # density: Hermite polynomials at 0; in the whitened form He_mu(0, I) = He_mu1(0) He_mu2(0)
    Dk = Diagonal(one(T) ./ sqrt.(diag(K2)))
    Si = Dk*inv(Dk*K2*Dk)*Dk                       # inverse insensitive to the units
    if white
        h0 = zeros(T, Nn + 1); h0[1] = one(T)
        for m in 2:2:Nn
            h0[m+1] = -(m - 1)*h0[m-1]
        end
        He = h0*h0'
    else
        He = hermite2(Nn, Si, zeros(T, 2))
    end
    logdet2 = sum(log.(diag(Lc)))                   # log det(K'')^(1/2)
    f0 = exp(K0 - dot(tau, x) - log(2*T(pi)) - logdet2)
    pdf = zeros(T, order); coef_err_pdf = fill(NaN, order); coef_err_cdf = fill(NaN, order)
    if G === nothing
        s = one(T); pdf[1] = f0; ds = zeros(T, ndraw); cerr && (coef_err_pdf[1] = 0.0)
        for M in 2:order
            m = M - 1
            for r in 1:2m
                k = 2m + 2r
                for n1 in 0:k
                    s += a[r][n1+1, k-n1+1]*He[n1+1, k-n1+1]
                    if cerr
                        for q in 1:ndraw
                            ds[q] += da[q][r][n1+1, k-n1+1]*He[n1+1, k-n1+1]
                        end
                    end
                end
            end
            pdf[M] = f0*s
            cerr && (coef_err_pdf[M] = Float64(maximum(abs.(ds))/abs(s)))
        end
    else
        # density of a pair of ratios (Corollary 1): coefficient of n^-m = sum_{k<=min(m,1)}
        # b_{m-k}(g_k), b_j(g_k) = sum_r sum_{|nu|+|mu|=2(j+r)} a_{r,nu} gamma_mu He_{nu+mu}(0),
        # gamma_mu the Taylor coefficients of g_k (in the whitened variable when white)
        if white
            if gamwhite !== nothing
                Gc = map(g -> padsq(T.(g), Nn + 1), Tuple(gamwhite(uh, Tm)))
                dG = map(g -> 2eps(T) .* abs.(g), Gc)
            else
                G0 = (padsq(T.(G[1]), Nn + 1), padsq(T.(G[2]), Nn + 1))
                Gc = (subs2(G0[1], Tm), subs2(G0[2], Tm))
                dG = map(g -> (degm .+ 2) .* eps(T) .* subs2(abs.(g), abs.(Tm)), G0)
            end
        else
            Gc = (padsq(T.(G[1]), Nn + 1), padsq(T.(G[2]), Nn + 1)); dG = (zero(Gc[1]), zero(Gc[2]))
        end
        dGq = [Tuple(dG[k] .* [sign(_rand(rng) - 0.5) for _ in 1:Nn+1, _ in 1:Nn+1] for k in 1:2) for _ in 1:ndraw]
        s = zero(T); ds = zeros(T, ndraw)
        for M in 1:order
            m = M - 1
            for k in 0:min(m, 1), r in 0:2(m-k)
                tot = 2(m - k + r)
                if r == 0
                    for m1 in 0:tot
                        s += Gc[k+1][m1+1, tot-m1+1]*He[m1+1, tot-m1+1]
                        if cerr
                            for q in 1:ndraw
                                ds[q] += dGq[q][k+1][m1+1, tot-m1+1]*He[m1+1, tot-m1+1]
                            end
                        end
                    end
                else
                    for l in 3r:tot, n1 in 0:l
                        anu = a[r][n1+1, l-n1+1]
                        for m1 in 0:tot-l
                            h = He[n1+m1+1, l-n1+tot-l-m1+1]
                            s += anu*Gc[k+1][m1+1, tot-l-m1+1]*h
                            if cerr
                                for q in 1:ndraw
                                    ds[q] += (da[q][r][n1+1, l-n1+1]*Gc[k+1][m1+1, tot-l-m1+1] +
                                              anu*dGq[q][k+1][m1+1, tot-l-m1+1])*h
                                end
                            end
                        end
                    end
                end
            end
            pdf[M] = f0*s
            cerr && (coef_err_pdf[M] = Float64(maximum(abs.(ds))/abs(s)))
        end
    end
    # orthant probability
    Fup = fill(T(NaN), order); canc = fill(T(NaN), order); kappa = fill(T(NaN), order)
    qerr = fill(T(NaN), order); nGu = 0
    status = maximum(x -> isnan(x) ? 0.0 : x, coef_err_pdf) > 1e-6 ? "coefficients inaccurate" : "ok"
    res() = (; pdf, Fup, canc, kappa, qerr, status, nG=nGu, coef_err_pdf, coef_err_cdf)
    if any(tau .< 0) && !direct                          # tau_j = 0 is allowed (limiting form)
        status = addstatus(status, "negative saddlepoint component (use spa_orthant2)")
        return res()
    end
    if T === Float64 && all(tau .>= 0) && K0 - dot(tau, x) - logscale < log(floatmin(T))
        # Chernoff bound P[X >= x] <= exp(K(tau) - tau'x): the (scaled) probability is below the
        # double-precision range, and is returned as zero, as an underflow
        Fup = zeros(T, order); qerr = zeros(T, order); canc = zeros(T, order); kappa = ones(T, order)
        status = addstatus(status, "probability below the double-precision range (use logscale)")
        return res()
    end
    L = 3p
    function osums(In, E)
        F = zeros(T, order); cn = zeros(T, order); kp = zeros(T, order); cf = fill(NaN, order)
        os_ = In[1, 1]; osa = abs(os_); om = 0; obig = zero(T); odd = zeros(T, ndraw)
        for oM in 1:order
            while om < 2oM - 1
                om += 1
                for orr in 1:om
                    ok_ = om + 2orr
                    for on1 in 0:ok_
                        oterm = a[orr][on1+1, ok_-on1+1]*In[on1+1, ok_-on1+1]
                        os_ += oterm; osa += abs(oterm); obig = max(obig, abs(oterm))
                        if cerr
                            for oq in 1:ndraw
                                odd[oq] += da[oq][orr][on1+1, ok_-on1+1]*In[on1+1, ok_-on1+1]
                            end
                        end
                    end
                end
            end
            F[oM] = E*os_; cn[oM] = obig/In[1, 1]; kp[oM] = osa/abs(os_)
            cerr && (cf[oM] = Float64(maximum(abs.(odd))/abs(os_)))
        end
        return F, cn, kp, cf
    end
    resolved(In) = isfinite(In[1, 1]) && In[1, 1] > 0
    if meth === :exact
        t = K2*tau
        E = exp(K0 - dot(tau, x) + dot(tau, K2*tau)/2 - T(logscale))
        Inu = Iexact(K2, inv(K2), t, tau, L)
        resolved(Inu) && ((Fup, canc, kappa, coef_err_cdf) = osums(Inu, E); qerr = zeros(T, order))
    else
        uu = tau ./ sqrt.([Si[1, 1], Si[2, 2]])          # tau_j/sd(W_j), W ~ N(0, Sigma^{-1})
        if all(uu .> 2.5)
            Inu, Ih = Ifreq(Tm, tau, L, 80)
            E = exp(K0 - dot(tau, x) - logscale - log(2pi) - logdet2)
            nGu = 80
            if resolved(Inu)
                Fup, canc, kappa, coef_err_cdf = osums(Inu, E)
                qerr = abs.(osums(Ih, E)[1] .- Fup) ./ abs.(Fup)
            end
        else
            # the exponential factor exp(K0 - tau'x + tau'K2 tau/2) and the normal density at the
            # nodes are combined in the log domain
            t = K2*tau
            logE = K0 - dot(tau, x) + dot(tau, K2*tau)/2 - logscale
            Gn = nG; Ih = nothing
            while true
                Inu = Ileg(K2, Lc, t, Gn, logE, L)
                resolved(Inu) || break
                Ih === nothing && (Ih = Ileg(K2, Lc, t, Gn ÷ 2, logE, L))
                Fup, canc, kappa, coef_err_cdf = osums(Inu, 1.0)
                qerr = abs.(osums(Ih, 1.0)[1] .- Fup) ./ abs.(Fup)
                (qerr[order] > qtol && 2Gn <= nGmax) || break
                Gn *= 2; Ih = Inu                        # escalate; the previous grid is the half grid
            end
            nGu = Gn
        end
    end
    if !resolved(Inu)                                    # the leading integral is not resolved
        Fup = fill(T(NaN), order); canc = fill(T(NaN), order); kappa = fill(T(NaN), order)
        qerr = fill(T(NaN), order); coef_err_cdf = fill(NaN, order)
        status = addstatus(status, "leading integral not resolved")
        return res()
    end
    maximum(x -> isnan(x) ? 0.0 : x, coef_err_cdf) > 1e-6 && !occursin("coefficients", status) &&
        (status = addstatus(status, "coefficients inaccurate"))
    qerr[order] <= qtol || (status = addstatus(status, "integration not resolved"))
    return res()
end

addstatus(s::AbstractString, msg::AbstractString) = s == "ok" ? msg : s * "; " * msg

# P padded with zeros, or truncated, to n x n
function padsq(P::AbstractMatrix{T}, n::Integer) where {T}
    Q = zeros(T, n, n); m1 = min(n, size(P, 1)); m2 = min(n, size(P, 2))
    Q[1:m1, 1:m2] .= P[1:m1, 1:m2]
    return Q
end

# coefficients of degree kmin..N of the product of the bivariate polynomials A and B (truncated)
function polyprod(A::AbstractMatrix{T}, B::AbstractMatrix{T}, kmin::Integer, N::Integer) where {T}
    C = zeros(T, N + 1, N + 1)
    for k in kmin:N, n1 in 0:k
        n2 = k - n1; s = zero(T)
        for j in 0:k, l1 in max(0, j - n2):min(j, n1)
            l2 = j - l1
            a = A[l1+1, l2+1]
            iszero(a) && continue
            s += a*B[n1-l1+1, n2-l2+1]
        end
        C[n1+1, n2+1] = s
    end
    return C
end

# whitened orthant integrals by Gauss-Legendre quadrature in conditioned coordinates (condgrid)
function Ileg(K2::Matrix{Float64}, Lc, t::Vector{Float64}, G::Integer, logE::Real, L::Integer)
    W1, W2, phiW = condgrid(K2, t, G, logE)
    V1 = W1 ./ Lc[1, 1]; V2 = (W2 .- Lc[2, 1] .* V1) ./ Lc[2, 2]     # v = L^{-1} w
    H1 = hermite1(V1, L); H2 = hermite1(V2, L)
    Inu = zeros(L + 1, L + 1)
    for k in 0:L, n1 in 0:k
        Inu[n1+1, k-n1+1] = sum(H1[n1+1] .* H2[k-n1+1] .* phiW)
    end
    return Inu
end

# coefficients of P(T u) for the bivariate polynomial P (P[n1+1,n2+1] = [s^(n1,n2)] P), T upper
# triangular: s1 = T11 u1 + T12 u2, s2 = T22 u2
function subs2(P::AbstractMatrix{T}, Tm::AbstractMatrix) where {T}
    n = size(P, 1); Q = zeros(T, n, n)
    T11 = T(Tm[1, 1]); T12 = T(Tm[1, 2]); T22 = T(Tm[2, 2])
    for n1 in 0:n-1, n2 in 0:n-1-n1
        c = P[n1+1, n2+1]
        iszero(c) && continue
        for j in 0:n1
            Q[j+1, n1-j+n2+1] += c*T(binomial(big(n1), j))*T11^j*T12^(n1-j)*T22^n2
        end
    end
    return Q
end

# H[m+1] = He_m(V), m = 0..L, by the three-term recursion
function hermite1(V::AbstractArray{T}, L::Integer) where {T}
    H = Vector{typeof(V)}(undef, L + 1)
    H[1] = ones(T, size(V)); L == 0 && return H
    H[2] = copy(V)
    for m in 2:L
        H[m+1] = V .* H[m] .- (m - 1) .* H[m-1]
    end
    return H
end

# He[n1+1,n2+1] = He_nu(x, Sigma), |nu| <= N, by the recursion of Takemura and Takeuchi (1988),
# Si = Sigma^{-1}.
function hermite2(N::Integer, Si::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    He = zeros(T, N + 1, N + 1); He[1, 1] = one(T)
    y = Si*x
    for k in 1:N, n1 in 0:k
        n2 = k - n1
        j, prev = n1 > 0 ? (1, (n1 - 1, n2)) : (2, (n1, n2 - 1))
        v = y[j]*He[prev[1]+1, prev[2]+1]
        if prev[1] > 0
            v -= Si[j, 1]*prev[1]*He[prev[1], prev[2]+1]
        end
        if prev[2] > 0
            v -= Si[j, 2]*prev[2]*He[prev[1]+1, prev[2]]
        end
        He[n1+1, n2+1] = v
    end
    return He
end


# Whitened form of I(nu)/phi(t,Sigma) = Re E[(iW)^nu/((tau1+iW1)(tau2+iW2))], W ~ N(0, Sigma^{-1}):
# Inu[mu1+1, mu2+1] = Re E[(iV)^mu/((tau1+iW1)(tau2+iW2))], W = T V, V standard normal,
# |mu| <= L, by G x G Gauss-Hermite quadrature, and Ih with G/2 nodes.
function Ifreq(Tm::AbstractMatrix, tau::Vector{Float64}, L::Integer, G::Integer)
    function run(n)
        x, w = gauher(n)
        Z1 = [x[i] for i in 1:n, j in 1:n]; Z2 = [x[j] for i in 1:n, j in 1:n]
        W1 = Tm[1, 1] .* Z1 .+ Tm[1, 2] .* Z2; W2 = Tm[2, 2] .* Z2
        q = [w[i]*w[j]/((tau[1] + im*W1[i, j])*(tau[2] + im*W2[i, j])) for i in 1:n, j in 1:n]
        P1 = Vector{Matrix{ComplexF64}}(undef, L + 1); P2 = similar(P1)
        P1[1] = ones(ComplexF64, n, n); P2[1] = ones(ComplexF64, n, n)
        for k in 1:L
            P1[k+1] = P1[k] .* (im .* Z1); P2[k+1] = P2[k] .* (im .* Z2)
        end
        I = zeros(L + 1, L + 1)
        for k in 0:L, n1 in 0:k
            I[n1+1, k-n1+1] = real(sum(q .* P1[n1+1] .* P2[k-n1+1]))
        end
        return I
    end
    return run(G), run(G ÷ 2)
end

# Quadrature grid for E int_{z >= t} h(z-t) phi(z,S) dz in conditioned coordinates: W1, W2 = z - t
# at the nodes (n1 x G), phiW = E phi(z,S) times the quadrature weights, with E = exp(logE)
# combined with phi in the log domain. The nodes in z1 cover 16 standard deviations on either side
# of the z1-coordinate m1 of the mode of phi(z,S) on {z >= t} (m1 = t1 when the saddlepoint is in
# the positive orthant, so that the range is [t1, t1 + 16 sd1]), split, with G nodes on each piece,
# at the edges of the layer in which the truncation of z2 given z1 at t2 is partial when that layer
# is narrow (correlations near +-1, layerbp) and where the density of z1 has decayed when t1 is far
# in its tail (tailbp); for each z1 the G nodes in z2 cover the part above t2 of an interval of 16
# conditional standard deviations around the conditional mean of z2 given z1.
function condgrid(S::Matrix{Float64}, t::Vector{Float64}, G::Integer, logE::Real=0.0)
    xg, wg = gauleg(G)
    sd1 = sqrt(S[1, 1]); b = S[1, 2]/S[1, 1]; s2 = sqrt(S[2, 2] - S[1, 2]^2/S[1, 1])
    m1 = modez1(S, t)
    a1 = max(t[1], m1 - 16sd1); b1 = m1 + 16sd1
    z1, w1 = glpieces(a1, b1, vcat(layerbp(a1, b1, b, 0.0, s2, t[2], 16), tailbp(a1, b1, 0.0, sd1, a1, 16)), xg, wg)
    n1 = length(z1)
    mu = b .* z1
    lo = max.(t[2], mu .- 16s2); hi = max.(mu, t[2]) .+ 16s2
    Z2 = [(lo[i] + hi[i])/2 + (hi[i] - lo[i])/2*xg[j] for i in 1:n1, j in 1:G]
    W2w = [(hi[i] - lo[i])/2*wg[j] for i in 1:n1, j in 1:G]
    phiW = [exp(logE - z1[i]^2/(2S[1, 1]) - log(sqrt(2pi)*sd1) -
                (Z2[i, j] - mu[i])^2/(2s2^2) - log(sqrt(2pi)*s2))*w1[i]*W2w[i, j]
            for i in 1:n1, j in 1:G]
    W1 = [z1[i] - t[1] for i in 1:n1, j in 1:G]
    W2 = Z2 .- t[2]
    return W1, W2, phiW
end

# Gauss-Legendre nodes and weights on [lo,hi], split at the breakpoints bp that lie inside (lo,hi),
# with the full number of nodes on each piece. Nothing is dropped: the split only adds nodes where
# the integrand changes on a scale much shorter than the interval (layerbp, tailbp). Without
# breakpoints the nodes are those of [lo,hi]. Generic in the floating-point type.
function glpieces(lo::T, hi::T, bp::AbstractVector, xg::AbstractVector, wg::AbstractVector) where {T}
    e = sort(T[v for v in bp if lo < v < hi])
    e = vcat(lo, e, hi)
    z = T[]; w = T[]
    for k in 1:length(e)-1
        e[k+1] > e[k] || continue
        append!(z, (e[k] + e[k+1])/2 .+ (e[k+1] - e[k])/2 .* xg)
        append!(w, (e[k+1] - e[k])/2 .* wg)
    end
    return z, w
end

# edges of the layer of the outer coordinate z in which a later coordinate, normal given z with
# mean icpt + slope z and standard deviation sd, has its truncation point tj within R sd of its mean
# (width 2 R sd/|slope|), when the layer is narrower than a quarter of [lo,hi], as when a
# correlation is near +-1; otherwise none
function layerbp(lo::T, hi::T, slope, icpt, sd, tj, R) where {T}
    (slope == 0 || !isfinite(slope)) && return T[]
    zA = T((tj - R*sd - icpt)/slope); zB = T((tj + R*sd - icpt)/slope)
    return abs(zB - zA) < (hi - lo)/4 ? T[zA, zB] : T[]
end

# for a coordinate N(m,s^2) truncated to [t,Inf) with t far above m, the point
# m + {(t - m)^2 + R^2 s^2}^(1/2) beyond which its density is below exp(-R^2/2) times that at t,
# when it is within a quarter of [lo,hi] of lo; otherwise none
function tailbp(lo::T, hi::T, m, s, t, R) where {T}
    t <= m && return T[]
    h = T(m + sqrt((t - m)^2 + R^2*s^2))
    return h - lo < (hi - lo)/4 ? T[h] : T[]
end

# first coordinate of the maximizer of -z'S^{-1}z/2 on {z >= t} (2 x 2): the vertex when
# S^{-1} t >= 0, otherwise the best point on the two edges or the origin
function modez1(S::Matrix{Float64}, t::Vector{Float64})
    Dk = Diagonal(1 ./ sqrt.(diag(S))); P = Dk*inv(Dk*S*Dk)*Dk
    all(P*t .>= 0) && return t[1]
    cand = Vector{Float64}[]
    all(t .<= 0) && push!(cand, [0.0, 0.0])
    push!(cand, [t[1], max(t[2], S[1, 2]/S[1, 1]*t[1])])       # edge z1 = t1
    push!(cand, [max(t[1], S[1, 2]/S[2, 2]*t[2]), t[2]])       # edge z2 = t2
    q = [dot(c, P*c) for c in cand]
    return cand[argmin(q)][1]
end

# Exact I(nu), |nu| <= L, from the binomial sum (14) and the recursion of Lemma 2 (for
# variable precision; the sums cancel in double precision for large t and |nu|).
function Iexact(S::Matrix{T}, Si::Matrix{T}, t::Vector{T}, tau::Vector{T}, L::Integer) where {T}
    Jm = Jtable(L, S, Si, t)
    Inu = zeros(T, L + 1, L + 1)
    for k in 0:L, n1 in 0:k
        n2 = k - n1
        v = zero(T)
        for m1 in 0:n1, m2 in 0:n2
            v += T(binomial(big(n1), m1)*binomial(big(n2), m2))*(-tau[1])^m1*(-tau[2])^m2*
                 Jm[n1-m1+1, n2-m2+1]
        end
        Inu[n1+1, n2+1] = v
    end
    return Inu
end

# Jm[r1+2, r2+2] = J(r) = prod_j (-D_j)^{r_j+1} Pbar(t,S), r_j >= -1, r1+r2 <= L-2:
# phi(t,S) He_r(t,S) for r >= 0 and Lemma 2 for r with a component -1.
function Jtable(L::Integer, S::Matrix{T}, Si::Matrix{T}, t::Vector{T}) where {T}
    Jm = zeros(T, L + 1, L + 1)
    He = hermite2(L, Si, t)
    phi2 = exp(-dot(t, Si*t)/2)/(2*T(pi)*sqrt(det(S)))
    for k in 0:L-1, n1 in 0:k
        Jm[n1+2, k-n1+2] = phi2*He[n1+1, k-n1+1]
    end
    for i in 1:2
        j = 3 - i
        Sjj = S[j, j]; Sij = S[i, j]
        cv = S[i, i] - Sij^2/Sjj
        g = exp(-t[j]^2/(2Sjj))/sqrt(2*T(pi)*Sjj)*erfc((t[i] - Sij/Sjj*t[j])/sqrt(2cv))/2
        Jr = zeros(T, L); Jr[1] = g                  # Jr[rj+1] = J(r), r_i = -1
        for rj in 0:L-2
            Jri = i == 1 ? Jm[2, rj+2] : Jm[rj+2, 2]
            v = t[j]/Sjj*Jr[rj+1] - Sij/Sjj*Jri
            rj > 0 && (v -= rj/Sjj*Jr[rj])
            Jr[rj+2] = v
        end
        if i == 1
            Jm[1, 2:L+1] = Jr
        else
            Jm[2:L+1, 1] = Jr
        end
    end
    Jm[1, 1] = orthant2(t, S)
    return Jm
end

"""
    orthant2(t, S)

``P(Z_1 ≥ t_1, Z_2 ≥ t_2)``, ``Z ∼ N(0,S)``, by one-dimensional adaptive quadrature in the
working precision.
"""
function orthant2(t::AbstractVector{T}, S::AbstractMatrix{T};
                  rtol=(T === Float64 ? 1e-14 : eps(T)^T(0.85))) where {T}
    sd1 = sqrt(S[1, 1]); sd2 = sqrt(S[2, 2]); r = S[1, 2]/(sd1*sd2)
    h = t[1]/sd1; k = t[2]/sd2
    c = sqrt(1 - r^2)
    f(z) = exp(-z^2/2)/sqrt(2*T(pi))*erfc((k - r*z)/(c*sqrt(T(2))))/2
    v, _ = quadgk(f, h, T(Inf); rtol=rtol, order=15, maxevals=10^6)
    return v
end

