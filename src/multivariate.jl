# Expansions in any dimension d: density (Theorem 3) and orthant probability (Theorem 1), with
# the inclusion-exclusion (17) for saddlepoints with negative components.
#
# Multi-indices nu with |nu| <= N are stored in graded order (degree 0, 1, 2, ...; within a degree
# with the first component descending, then recursively), and located by mi_rank.

# all multi-indices of dimension d with |nu| <= N, in graded order
mi_list(d::Integer, N::Integer) = reduce(vcat, [compositions(k, d) for k in 0:N])

struct MIndex{d}
    N::Int
    list::Vector{NTuple{d,Int}}
    B::Matrix{Int}                 # B[n+1, k+1] = binomial(n, k)
end
function MIndex(d::Integer, N::Integer)
    nb = N + d + 1
    B = [binomial(n, k) for n in 0:nb, k in 0:nb]
    MIndex{d}(N, mi_list(d, N), B)
end
_binom(I::MIndex, n, k) = (n < 0 || k < 0 || k > n) ? 0 : I.B[n+1, k+1]
# 1-based position of nu in I.list
function mi_rank(I::MIndex{d}, nu) where {d}
    k = sum(nu)
    r = k == 0 ? 0 : _binom(I, k - 1 + d, d)
    ki = k
    for i in 1:d-1
        r += _binom(I, ki - nu[i] - 1 + d - i, d - i)
        ki -= nu[i]
    end
    return r + 1
end
mi_count(I::MIndex{d}, N::Integer) where {d} = _binom(I, N + d, d)    # number with |nu| <= N


"""
    spa_mv(Kfun, x, order; tau0=nothing, method=:auto, refine=true, nG=64, nGH=48, nqmc=2^14,
           nshift=8, seed=1, Kwhite=nothing, qtol=nothing, nGmax=nothing)
        -> (; pdf, Fup, tau, kappa, err, kappa_pdf, refined, info)

Saddlepoint approximations of orders `1:order` in dimension `d = length(x)` of the density
(Theorem 3) and of the orthant probability ``P(X ≥ x)`` (Theorem 1). `Kfun(nu, t)` returns the
derivative of order `nu` (a `d`-tuple of integers) of the cumulant generating function at `t`
(`NaN` outside the domain).

* `pdf[M]`: density expansion with the terms `m ≤ M-1`, a finite sum evaluated without integrals
  (not the exact density);
* `Fup[M]`: ``P(X ≥ x)`` with the terms `m ≤ 2M-1`, when the saddlepoint `tau` has no negative
  component (`NaN` otherwise; use [`spa_orthant`](@ref));
* `kappa[M]`, `kappa_pdf[M]`: cancellation indicators of the final sums `Fup[M]` and `pdf[M]`
  (sum of absolute values of the terms over the absolute value of the sum); they do not measure
  errors in the coefficients (see `coef_err_pdf`, `coef_err_cdf`);
* `err[M]`: error estimate of `Fup[M]` from the integrals ``I(ν)``, for every order: the change
  with half the quadrature nodes (`:quad`) or the standard error over random shifts (`:qmc`);
  `NaN` when no estimate is available (`:exact`, whose base probabilities come from adaptive
  quadrature). These compare two quadratures, are not bounds, and concern the evaluation of the
  retained terms only, not the asymptotic error;
* `info`: named tuple with `kappa_pdf`; `method_cdf`, the method actually used for `Fup`;
  `refined_pdf`, `refined_cdf`, `refined`, `digits` and `precision` (the working precision of the
  returned values; see `refine`); `tau_refined`; `saddle` (see [`spa_saddle`](@ref)); `nodes` (the
  quadrature or QMC settings actually used); `coef_err_pdf[M]`, `coef_err_cdf[M]`: estimates, in
  double precision, of the relative errors of `pdf[M]` and `Fup[M]` caused by rounding in the
  formation of the whitened Taylor coefficients (first-order propagation of random perturbations
  of the size of the rounding errors of the substitution; with `Kwhite` only the rounding of the
  supplied derivatives is included; `NaN` in `BigFloat`); and `status`: `"ok"` or one or more of
  (joined by `"; "`) `"negative saddlepoint component (use spa_orthant)"`, `"leading integral not
  resolved"` (`Fup` and `err` are then `NaN`), `"integration not resolved"` (``err[order] >``
  `qtol` ``|Fup[order]|`` after the node escalation), `"coefficients inaccurate"` (a
  coefficient-error estimate above ``10^{-6}`` that no refinement removed), `"probability below
  the double-precision range"` (the Chernoff bound ``e^{K(τ)-τ'x}`` is below `floatmin`; `Fup` is
  then zero), `"no orthant probability requested"`, `"refinement failed"` or `"no error
  estimate"`.

The integrals ``I(ν)`` of the orthant probability are computed by `method`:
* `:quad` (`d = 3`, default there): Gauss-Hermite quadrature of the representation of Appendix B
  with `nGH` nodes per coordinate when ``τ_j\\{(Σ^{-1})_{jj}\\}^{-1/2} > 2.5`` for all `j`, and
  otherwise Gauss-Legendre quadrature of the orthant-integral form with `nG` nodes per coordinate
  and piece in conditioned coordinates, the ranges of ``z_1`` and ``z_2`` split at the boundary
  layers created by correlations near ``±1`` and at fast tail decays; while ``err[order] >``
  `qtol` ``|Fup[order]|`` (`qtol` default ``10^{-8}``, ``10^{-3}`` for `:qmc`) the number of nodes
  is doubled, up to `nGmax` (default `2nG`);
* `:qmc` (default for `d ≥ 4`): randomized quasi-Monte Carlo (Richtmyer lattice, antithetic,
  `nshift` random shifts of `nqmc` points) after Genz's sequential conditioning of the truncated
  normal;
* `:exact` (`d = 3`, default for `BigFloat`; also `:recursion`): binomial sum (14) and recursion
  of Lemma 2 in the working precision, with the normal orthant probabilities of dimensions two and
  three by adaptive quadrature in the working precision (so more digits do not by themselves make
  them more accurate).

In the double-precision methods the exponential factor ``E_n`` of (12) is combined with the normal
density or the probabilities of the sequential conditioning in the log domain, and the density
prefactor is formed from the logarithms of the diagonal of the Cholesky factor of ``K''``, so that
the products are formed without overflow or underflow of their factors.

Whitened evaluation. With ``K''=LL'`` (Cholesky) and ``T=L^{-T}``, the terms of each order are
evaluated for ``Y=L^{-1}X``: the Taylor coefficients of ``K_Y(u)=K(Tu)`` at ``L'τ`` replace those of
``K`` at ``τ``, the Hermite polynomials ``He_ν(w,K'')`` are replaced by products of univariate ones
in ``L^{-1}w``, and the integrals ``I(ν)`` by integrals of these products over the same orthant, at
the same nodes (in the frequency-domain representation ``(iW)^ν`` becomes ``(iV)^μ``, ``W=TV``).
The sums of the terms of a given order are unchanged in exact arithmetic, and the terms cancel much
less when the components are strongly correlated under the tilted distribution, provided the
whitened coefficients are formed accurately. `Kwhite` is an optional function `(nu, u, T) ->`
derivative of order `nu` at `u` of ``K_Y(u)=K(Tu)``, the cumulant generating function of ``Y=T'X``,
for a ``d×d`` matrix `T`; without it the Taylor coefficients of ``K_Y`` are obtained by polynomial
substitution, whose loss of accuracy `coef_err` estimates. The method `:exact` keeps the original coordinates, since
the recursion of Lemma 2 is tied to the orthant.

`refine` (default `true`): in more than two dimensions the terms of both expansions can cancel by
many orders of magnitude at high orders, and the substitution can lose digits; when a
double-precision cancellation indicator exceeds ``10^6``, or a coefficient-error estimate exceeds
``10^{-10}``, the computation is repeated in `BigFloat` with about ``\\log_{10}κ+26`` digits (``κ``
replaced by the estimate divided by `eps()` when that is larger), the derivatives of ``K`` and
hence the coefficients being computed in that precision: the density always, and for `d = 3` with the quadrature method also the orthant probability (`:exact`). If
this fails (for instance because `Kfun` does not accept `BigFloat` arguments), the
double-precision values are kept, with a warning and `info.status = "refinement failed"`.
"""
function spa_mv(Kfun, x::AbstractVector, order::Integer; refine::Bool=true, method::Symbol=:auto,
                kwargs...)
    isempty(x) && throw(ArgumentError("spa_mv: x must be a nonempty vector"))
    all(isfinite, x) || throw(ArgumentError("spa_mv: x must be finite"))
    order >= 1 || throw(ArgumentError("spa_mv: order must be a positive integer"))
    method === :recursion && (method = :exact)
    method in (:auto, :quad, :qmc, :exact, :none) ||
        throw(ArgumentError("spa_mv: unknown method $method"))
    d = length(x)
    r = _spa_mv(Kfun, x, order; method, kwargs...)
    T = eltype(r.pdf)
    prec = T === Float64 ? "double" : "BigFloat ($(precision(T)) bits)"
    info = (; kappa_pdf=r.kappa_pdf, method_cdf=r.method_cdf, refined_pdf=false,
            refined_cdf=false, refined=false, digits=NaN, precision=prec, tau_refined=nothing,
            saddle=r.saddle, nodes=r.nodes, coef_err_pdf=r.coef_err_pdf,
            coef_err_cdf=r.coef_err_cdf, status=r.status)
    out(pdf, Fup, kappa, err, kappa_pdf, info) =
        (; pdf, Fup, tau=r.tau, kappa, err, kappa_pdf, refined=info.refined, info)
    nz(v) = map(z -> isnan(z) ? 0.0 : z, v)
    cflag(v) = any(nz(v) .> 1e-6)                          # coefficient-error flag
    function finish(info, fp, fc)
        (fp || fc) || return info
        return merge(info, (; status=addstatus(info.status, "coefficients inaccurate")))
    end
    if !(refine && T === Float64)
        info = finish(info, cflag(r.coef_err_pdf), cflag(r.coef_err_cdf))
        return out(r.pdf, r.Fup, r.kappa, r.err, r.kappa_pdf, info)
    end
    # refinement: cancellation in the sums (kappa) or loss of accuracy in the formation of the
    # whitened coefficients (coef_err) beyond about 1e-10 relative
    cp = nz(r.coef_err_pdf); cc = nz(r.coef_err_cdf)
    needpdf = !all(r.kappa_pdf .<= 1e6) || any(cp .> 1e-10)       # NaN or Inf counts as needing it
    needcdf = d == 3 && method in (:auto, :quad) && all(r.tau .>= 0) &&
              r.status != "leading integral not resolved" && (!all(r.kappa .<= 1e6) || any(cc .> 1e-10))
    if !(needpdf || needcdf)
        info = finish(info, cflag(r.coef_err_pdf), cflag(r.coef_err_cdf))
        return out(r.pdf, r.Fup, r.kappa, r.err, r.kappa_pdf, info)
    end
    kap = max(maximum(r.kappa_pdf), maximum(cp)/eps())
    needcdf && (kap = max(kap, maximum(r.kappa), maximum(cc)/eps()))
    kap < 1e30 || (kap = 1e30)                             # cap the precision
    dig = ceil(Int, log10(kap) + 26)
    bits = ceil(Int, dig*log2(10))
    rb = try
        setprecision(BigFloat, bits) do
            kw = Dict{Symbol,Any}(kwargs); delete!(kw, :tau0)
            _spa_mv(Kfun, big.(x), order; tau0=big.(r.tau), method=(needcdf ? :exact : :none), kw...)
        end
    catch err
        err isa InterruptException && rethrow()
        @warn "spa_mv: refinement in BigFloat failed ($(sprint(showerror, err))); double-precision values returned"
        nothing
    end
    if rb === nothing
        info = merge(info, (; status="refinement failed"))
        return out(r.pdf, r.Fup, r.kappa, r.err, r.kappa_pdf, info)
    end
    pdf = Float64.(rb.pdf); kappa_pdf = Float64.(rb.kappa_pdf)
    Fup, kappa, err = r.Fup, r.kappa, r.err
    info = merge(info, (; kappa_pdf, refined_pdf=true, refined=true, digits=dig,
                        precision="BigFloat ($bits bits)", tau_refined=Float64.(rb.tau)))
    if needcdf
        Fup = Float64.(rb.Fup); kappa = Float64.(rb.kappa); err = fill(NaN, order)
        info = merge(info, (; refined_cdf=true, method_cdf=:exact, status="no error estimate"))
    end
    info = finish(info, false, cflag(r.coef_err_cdf) && !info.refined_cdf && !any(isnan, Fup))
    return out(pdf, Fup, kappa, err, kappa_pdf, info)
end

function _spa_mv(Kfun, x::AbstractVector, order::Integer; tau0=nothing, method::Symbol=:auto,
                 nG::Integer=64, nGH::Integer=48, nqmc::Integer=2^14, nshift::Integer=8,
                 seed::Integer=1, Kwhite=nothing, qtol=nothing, nGmax=nothing)
    d = length(x)
    T = float(tau0 === nothing ? eltype(x) : promote_type(eltype(x), eltype(tau0)))
    x = T.(collect(x))
    t0 = tau0 === nothing ? zeros(T, d) : T.(collect(tau0))
    tau, sinfo = spa_saddle(Kfun, x, t0)
    p = 2order - 1
    Kd = Dict{NTuple{d,Int},T}()
    for nu in mi_list(d, p + 2)
        Kd[nu] = Kfun(nu, tau)
    end
    r = spa_mv_at(Kd, tau, x, order; method, nG, nGH, nqmc, nshift, seed, Kwhite, qtol, nGmax)
    return (; r..., saddle=sinfo)
end

"""
    spa_tri(Kfun, x, order; kwargs...)

Trivariate version of [`spa_mv`](@ref) (`length(x) == 3`): the integrals ``I(ν)`` of the orthant
probability by deterministic quadrature in double precision (`:quad`, whitened coordinates) and
by the recursion of Lemma 2 in `BigFloat` (`:exact`), with the automatic refinement in `BigFloat`
described under `spa_mv`. For saddlepoints with negative components use [`spa_orthant`](@ref).
"""
function spa_tri(Kfun, x::AbstractVector, order::Integer; kwargs...)
    length(x) == 3 || throw(ArgumentError("spa_tri: x must have three components"))
    return spa_mv(Kfun, x, order; kwargs...)
end

"""
    spa_mv_at(Kd, tau, x, order; method=:auto, Kwhite=nothing, ...)
        -> (; pdf, Fup, tau, kappa, err, kappa_pdf, method_cdf, nodes, status)

The expansions of [`spa_mv`](@ref) from the derivatives at a given saddlepoint: `Kd` is a `Dict`
from multi-indices (tuples) to ``K^{(ν)}(τ)``, ``|ν| ≤ 2\\,\\mathrm{order}+1``.
"""
function spa_mv_at(Kd::AbstractDict, tau::AbstractVector, x::AbstractVector, order::Integer;
                   method::Symbol=:auto, nG::Integer=64, nGH::Integer=48, nqmc::Integer=2^14,
                   nshift::Integer=8, seed::Integer=1, Kwhite=nothing, qtol=nothing, nGmax=nothing)
    d = length(x)
    T = float(promote_type(eltype(tau), eltype(x), valtype(Kd)))
    tau = T.(collect(tau)); x = T.(collect(x))
    method === :recursion && (method = :exact)
    meth = method === :auto ? (T === Float64 ? (d == 3 ? :quad : :qmc) : :exact) : method
    p = 2order - 1
    Nn = max(3p, 6(order - 1))
    I = MIndex(d, Nn)
    nN = length(I.list)
    deg = [sum(nu) for nu in I.list]
    e(j) = ntuple(i -> i == j ? 1 : 0, d)
    K0 = T(Kd[ntuple(_ -> 0, d)])
    K2 = [T(Kd[e(i) .+ e(j)]) for i in 1:d, j in 1:d]
    # theta_nu = K_nu/nu!, 3 <= |nu| <= p+2, of K at tau (theta0) and of K_Y(u) = K(T u) at
    # uh = L' tau (theta), K2 = L L', T = L^{-T};  a_r = [u^nu] G^r/r! for K_Y
    Lc = cholesky(Symmetric(K2)).L
    Tm = inv(Matrix(Lc')); uh = Lc'*tau
    logdet2 = sum(log.(diag(Lc)))                        # log det(K'')^(1/2)
    white = meth !== :exact
    cerr = T === Float64                                 # coefficient-error estimates
    sel = findall(k -> 3 <= deg[k] <= p + 2, 1:nN)
    fsel = [prod(factT(T, n) for n in I.list[k]) for k in sel]
    theta0 = zeros(T, nN)
    for (i, k) in enumerate(sel)
        theta0[k] = T(Kd[I.list[k]])/fsel[i]
    end
    dth = zeros(T, nN)
    if !white
        theta = theta0                                   # Lemma 2 works in the original coordinates
        cerr && (dth = 2eps(T) .* abs.(theta0))
    elseif Kwhite !== nothing
        theta = zeros(T, nN)
        for (i, k) in enumerate(sel)
            theta[k] = T(Kwhite(I.list[k], uh, Tm))/fsel[i]
        end
        cerr && (dth = 2eps(T) .* abs.(theta))
    else
        theta = polysub_mv(theta0, Tm, I)
        cerr && (dth = (deg .+ 2) .* eps(T) .* polysub_mv(abs.(theta0), abs.(Tm), I))
    end
    supp = [(I.list[k], theta[k]) for k in sel if !iszero(theta[k])]
    R = max(p, 2(order - 1))
    function mulpoly(prev, terms, r)                     # [u^nu] of prev times sum(terms)
        A = zeros(T, nN)
        for (kidx, ka) in enumerate(I.list)
            v = prev[kidx]
            (iszero(v) || deg[kidx] < 3(r - 1)) && continue
            for (mu, th) in terms
                deg[kidx] + sum(mu) > Nn && continue
                A[mi_rank(I, ka .+ mu)] += th*v
            end
        end
        return A
    end
    a = Vector{Vector{T}}(undef, max(R, 1))
    a[1] = theta
    for r in 2:R
        a[r] = mulpoly(a[r-1], supp, r) ./ r
    end
    # first-order propagation of random perturbations of the whitened coefficients of the size of
    # their rounding errors (double precision): da[r] = a[r-1] dtheta, the derivative of G^r/r!
    ndraw = 2
    da = [Vector{Vector{T}}(undef, max(R, 1)) for _ in 1:ndraw]
    if cerr
        rng = _Lcg(1)
        for q in 1:ndraw
            dt = dth .* [sign(_rand(rng) - 0.5) for _ in 1:nN]
            da[q][1] = dt
            dsupp = [(I.list[k], dt[k]) for k in 1:nN if !iszero(dt[k])]
            for r in 2:R
                da[q][r] = mulpoly(a[r-1], dsupp, r)
            end
        end
    end
    # density: Hermite polynomials at 0, in the whitened coordinates products of univariate values
    Dk = Diagonal(one(T) ./ sqrt.(diag(K2)))
    Si = Dk*inv(Dk*K2*Dk)*Dk                             # inverse insensitive to the units
    SiH = white ? Matrix{T}(LinearAlgebra.I, d, d) : Si
    He0 = hermite_mv(I, SiH, zeros(T, d, 1))[1, :]
    f0 = exp(K0 - dot(tau, x) - (T(d)/2)*log(2*T(pi)) - logdet2)
    pdf = zeros(T, order); kappa_pdf = ones(T, order); s = one(T); sa = one(T); pdf[1] = f0
    coef_err_pdf = fill(NaN, order); coef_err_cdf = fill(NaN, order)
    dsp = zeros(T, ndraw); cerr && (coef_err_pdf[1] = 0.0)
    for M in 2:order
        m = M - 1
        for r in 1:2m, k in findall(==(2m + 2r), deg)
            term = a[r][k]*He0[k]
            s += term; sa += abs(term)
            if cerr
                for q in 1:ndraw
                    dsp[q] += da[q][r][k]*He0[k]
                end
            end
        end
        pdf[M] = f0*s; kappa_pdf[M] = sa/abs(s)
        cerr && (coef_err_pdf[M] = Float64(maximum(abs.(dsp))/abs(s)))
    end
    # orthant probability
    Fup = fill(T(NaN), order); kappa = fill(T(NaN), order); err = fill(T(NaN), order)
    nodes = (; method=meth, nG=NaN, nGH=NaN, nqmc=NaN, nshift=NaN)
    res(status) = (; pdf, Fup, tau, kappa, err, kappa_pdf, method_cdf=meth, nodes, status,
                   coef_err_pdf, coef_err_cdf)
    meth === :none && return res("no orthant probability requested")
    any(tau .< 0) && return res("negative saddlepoint component (use spa_orthant)")
    if T === Float64 && K0 - dot(tau, x) < log(floatmin(T))
        # Chernoff bound P[X >= x] <= exp(K(tau) - tau'x): the probability is below the double-
        # precision range, and is returned as zero, as an underflow
        Fup = zeros(T, order); err = zeros(T, order); kappa = ones(T, order)
        return res("probability below the double-precision range")
    end
    L = 3p
    IL = MIndex(d, L)
    t = K2*tau
    uu = Float64.(tau ./ sqrt.(diag(Si)))
    qt = qtol === nothing ? (meth === :qmc ? 1e-3 : 1e-8) : qtol
    degL = [sum(nu) for nu in IL.list]
    rk = [mi_rank(I, nu) for nu in IL.list]               # position of nu in the list of a
    idxk = [findall(==(k), degL) for k in 1:L]
    E = one(T)
    function sums(Inu)
        F = zeros(T, order); ka = zeros(T, order); cf = fill(NaN, order)
        ss = T(Inu[1]); sb = abs(ss); mm = 0; dd = zeros(T, ndraw)
        for MM in 1:order
            while mm < 2MM - 1
                mm += 1
                for rr in 1:mm, kk in idxk[mm+2rr]
                    tt = a[rr][rk[kk]]*Inu[kk]
                    ss += tt; sb += abs(tt)
                    if cerr
                        for qq in 1:ndraw
                            dd[qq] += da[qq][rr][rk[kk]]*Inu[kk]
                        end
                    end
                end
            end
            F[MM] = E*ss; ka[MM] = sb/abs(ss)
            cerr && (cf[MM] = Float64(maximum(abs.(dd))/abs(ss)))
        end
        return F, ka, cf
    end
    resolved(v) = isfinite(v[1]) && v[1] > 0             # tested in the working precision
    status = "ok"
    if meth === :quad
        (d == 3 && T === Float64) || throw(ArgumentError("spa_mv: method :quad needs d = 3 and Float64"))
        if all(uu .> 2.5)
            Inu, Ih = Ifreq_mv(IL, Tm, tau, nGH)          # whitened I(nu)/phi(t,Sigma)
            E = exp(K0 - dot(tau, x) - (d/2)*log(2pi) - logdet2)
            nodes = merge(nodes, (; nGH))
            if resolved(Inu)
                Fup, kappa, coef_err_cdf = sums(Inu); err = abs.(sums(Ih)[1] .- Fup)
            end
        else
            logE = K0 - dot(tau, x) + dot(tau, K2*tau)/2  # the exponential factor is in the weights
            Gmax = nGmax === nothing ? 2nG : nGmax
            Gn = nG; Ih = nothing
            local Inu
            while true
                Inu = Iquad3(IL, K2, Matrix(Lc), t, Gn, logE)
                resolved(Inu) || break
                Ih === nothing && (Ih = Iquad3(IL, K2, Matrix(Lc), t, Gn ÷ 2, logE))
                Fup, kappa, coef_err_cdf = sums(Inu); err = abs.(sums(Ih)[1] .- Fup)
                (err[order] > qt*abs(Fup[order]) && 2Gn <= Gmax) || break
                Gn *= 2; Ih = Inu                         # escalate; the previous grid is the half grid
            end
            nodes = merge(nodes, (; nG=Gn))
        end
        Isets = [Inu]
    elseif meth === :qmc
        T === Float64 || throw(ArgumentError("spa_mv: method :qmc needs Float64"))
        Isets = Iqmc(IL, K2, Matrix(Lc), t, nqmc, nshift, seed, K0 - dot(tau, x) + dot(tau, K2*tau)/2)
        nodes = merge(nodes, (; nqmc, nshift))
        if all(resolved, Isets)
            Fs = [sums(Is) for Is in Isets]
            Fall = reduce(hcat, first.(Fs))
            Fup = vec(sum(Fall; dims=2)) ./ length(Fs)
            err = vec(sqrt.(sum((Fall .- Fup).^2; dims=2) ./ (length(Fs)*(length(Fs) - 1))))
            _, kappa, coef_err_cdf = sums(sum(Isets) ./ length(Isets))
        end
    elseif meth === :exact
        d == 3 || throw(ArgumentError("spa_mv: method :exact is implemented for d = 3"))
        Isets = [Iexact3(IL, K2, inv(K2), t, tau)]
        E = exp(K0 - dot(tau, x) + dot(tau, K2*tau)/2)
        if resolved(Isets[1])
            Fup, kappa, coef_err_cdf = sums(Isets[1])
            status = "no error estimate"                  # adaptive quadrature of the base probabilities
        end
    else
        throw(ArgumentError("spa_mv: method must be :auto, :quad, :qmc, :exact or :none"))
    end
    if !all(resolved, Isets)                              # the leading integral is not resolved
        Fup = fill(T(NaN), order); kappa = fill(T(NaN), order); err = fill(T(NaN), order)
        coef_err_cdf = fill(NaN, order)
        return res("leading integral not resolved")
    end
    meth !== :exact && !(err[order] <= qt*abs(Fup[order])) && (status = addstatus(status, "integration not resolved"))
    return res(status)
end

# coefficients of P(T u) for the polynomial P with coefficients th0 on the multi-index list of I,
# T upper triangular: the variables are substituted slot by slot, s_j -> sum_{k>=j} T[j,k] u_k,
# for j = d, d-1, ..., 1, so that the u_k already introduced (k > j) are not transformed again
function polysub_mv(th0::Vector{T}, Tm::AbstractMatrix, I::MIndex{d}) where {T,d}
    th = copy(th0)
    for j in d:-1:1
        new = zeros(T, length(th))
        for q in findall(!iszero, th)
            nu = I.list[q]; m = nu[j]
            if m == 0
                new[q] += th[q]; continue
            end
            for c in compositions(m, d - j + 1)          # compositions of m into the slots j..d
                mu = collect(nu); mu[j] -= m
                coef = T(factorial(big(m)))
                for (i, ci) in enumerate(c)
                    mu[j+i-1] += ci
                    coef *= T(Tm[j, j+i-1])^ci/T(factorial(big(ci)))
                end
                new[mi_rank(I, Tuple(mu))] += th[q]*coef
            end
        end
        th = new
    end
    return th
end

# He[j, k] = He_{nu_k}(W[:, j], Sigma), nu_k = I.list[k], by the recursion of Takemura and
# Takeuchi (1988): He_{nu} = y_i He_{nu-e_i} - sum_l Si[i,l] (nu-e_i)_l He_{nu-e_i-e_l},
# i the first nonzero component of nu, y = Si w (one column per multi-index).
function hermite_mv(I::MIndex{d}, Si::AbstractMatrix{T}, W::AbstractMatrix{T}) where {d,T}
    nN = length(I.list); n = size(W, 2)
    Y = permutedims(Si*W)                      # n x d
    He = zeros(T, n, nN)
    He[:, 1] .= one(T)
    for (k, nu) in enumerate(I.list)
        k == 1 && continue
        i = findfirst(>(0), nu)
        pr = Base.setindex(nu, nu[i] - 1, i)
        kp = mi_rank(I, pr)
        @views He[:, k] .= Y[:, i] .* He[:, kp]
        for l in 1:d
            pr[l] > 0 || continue
            kpp = mi_rank(I, Base.setindex(pr, pr[l] - 1, l))
            c = Si[i, l]*pr[l]
            @views He[:, k] .-= c .* He[:, kpp]
        end
    end
    return He
end

# I(nu) = sum_n wts[n] He_nu(W[:, n]) over chunks of nodes, |nu| <= IL.N
function Isum(IL::MIndex{d}, Si::Matrix{Float64}, W::Matrix{Float64}, wts::Vector{Float64}) where {d}
    nN = length(IL.list); n = size(W, 2)
    chunk = max(256, min(n, floor(Int, 1e7/nN)))
    Inu = zeros(nN)
    for c0 in 1:chunk:n
        c1 = min(n, c0 + chunk - 1)
        He = hermite_mv(IL, Si, W[:, c0:c1])
        Inu .+= He'*wts[c0:c1]
    end
    return Inu
end


# ----- the integrals I(nu), whitened: Inu[k] = sum of the weights times He_{nu_k}(L^{-1}(z - t), I)

# d = 3: Gauss-Legendre quadrature of the orthant-integral form in conditioned coordinates, G
# nodes per coordinate and piece; the Hermite polynomials are the products of univariate ones in the
# whitened variable v = L^{-1}(z - t). The weights include the exponential factor exp(logE),
# combined with the normal density in the log domain. Nodes z1 in [t1, t1 + 16 sd1], z2 | z1 and
# z3 | z1, z2 over the parts above t2 and t3 of intervals of 16 conditional standard deviations
# around the conditional means; the ranges of z1 and z2 are split at narrow boundary layers created
# by the truncation of the later coordinates (layerbp), which resolves correlations near +-1, and
# where their densities have decayed when the truncation point is far in the tail (tailbp).
function Iquad3(IL::MIndex{3}, S::Matrix{Float64}, Lc::Matrix{Float64}, t::Vector{Float64},
                G::Integer, logE::Real)
    I3 = Matrix{Float64}(LinearAlgebra.I, 3, 3)
    xg, wg = gauleg(G)
    sd1 = sqrt(S[1, 1]); b21 = S[2, 1]/S[1, 1]; s2 = sqrt(S[2, 2] - S[2, 1]^2/S[1, 1])
    c31 = S[3, 1]/S[1, 1]; s31 = sqrt(S[3, 3] - S[3, 1]^2/S[1, 1])
    S12 = S[1:2, 1:2]; beta = S[3, 1:2]' / S12; s3 = sqrt(S[3, 3] - dot(beta, S[1:2, 3]))
    lo1 = t[1]; hi1 = t[1] + 16sd1
    z1s, w1s = glpieces(lo1, hi1, vcat(layerbp(lo1, hi1, b21, 0.0, s2, t[2], 16),
                                       layerbp(lo1, hi1, c31, 0.0, s31, t[3], 16),
                                       tailbp(lo1, hi1, 0.0, sd1, t[1], 16)), xg, wg)
    P1 = Float64[]; P2 = Float64[]; LW = Float64[]          # pairs (z1, z2) and their log weights
    for (z1, w1) in zip(z1s, w1s)
        lw1 = log(w1) - z1^2/(2S[1, 1]) - log(sqrt(2pi)*sd1)
        mu2 = b21*z1; lo2 = max(t[2], mu2 - 16s2); hi2 = max(mu2, t[2]) + 16s2
        z2s, w2s = glpieces(lo2, hi2, vcat(layerbp(lo2, hi2, beta[2], beta[1]*z1, s3, t[3], 16),
                                           tailbp(lo2, hi2, mu2, s2, t[2], 16)), xg, wg)
        for (z2, w2) in zip(z2s, w2s)
            push!(P1, z1); push!(P2, z2)
            push!(LW, lw1 + log(w2) - (z2 - mu2)^2/(2s2^2) - log(sqrt(2pi)*s2))
        end
    end
    n12 = length(P1)
    n12 == 0 && return zeros(length(IL.list))
    W = zeros(3, n12*G); wc = zeros(n12*G); idx = 0
    for p in 1:n12
        z1 = P1[p]; z2 = P2[p]
        mu3 = beta[1]*z1 + beta[2]*z2; lo3 = max(t[3], mu3 - 16s3); hi3 = max(mu3, t[3]) + 16s3
        for k in 1:G
            z3 = (lo3 + hi3)/2 + (hi3 - lo3)/2*xg[k]
            lw3 = log((hi3 - lo3)/2*wg[k]) - (z3 - mu3)^2/(2s3^2) - log(sqrt(2pi)*s3)
            idx += 1
            W[1, idx] = z1 - t[1]; W[2, idx] = z2 - t[2]; W[3, idx] = z3 - t[3]
            wc[idx] = exp(logE + LW[p] + lw3)
        end
    end
    return Isum(IL, I3, LowerTriangular(Lc) \ W, wc)
end

# whitened form of I(nu)/phi(t,Sigma) = Re E[(iW)^nu/prod_j(tau_j+iW_j)], W ~ N(0, Sigma^{-1}):
# Re E[(iV)^mu/prod_j(tau_j+iW_j)] with W = T V, V standard normal, by G^d Gauss-Hermite
# quadrature, and with G/2 nodes
function Ifreq_mv(IL::MIndex{d}, Tm::AbstractMatrix, tau::Vector{Float64}, G::Integer) where {d}
    C = Matrix{Float64}(Tm)
    function run(G)
        xg, wg = gauher(G)
        nN = length(IL.list)
        Inu = zeros(nN)
        idxs = vec(CartesianIndices(ntuple(_ -> G, d)))
        chunk = max(256, floor(Int, 5e6/nN))
        for c0 in 1:chunk:length(idxs)
            c1 = min(length(idxs), c0 + chunk - 1)
            n = c1 - c0 + 1
            Z = zeros(d, n); q = ones(ComplexF64, n)
            for (jj, ci) in enumerate(idxs[c0:c1])
                for j in 1:d
                    Z[j, jj] = xg[ci[j]]; q[jj] *= wg[ci[j]]
                end
            end
            W = C*Z
            for jj in 1:n, j in 1:d
                q[jj] /= (tau[j] + im*W[j, jj])
            end
            V = zeros(ComplexF64, n, nN); V[:, 1] .= 1
            iZ = permutedims(im .* Z)
            for (k, nu) in enumerate(IL.list)
                k == 1 && continue
                i = findfirst(>(0), nu)
                kp = mi_rank(IL, Base.setindex(nu, nu[i] - 1, i))
                @views V[:, k] .= V[:, kp] .* iZ[:, i]
            end
            Inu .+= real.(transpose(V)*q)
        end
        Inu
    end
    return run(G), run(G ÷ 2)
end

# randomized QMC: Richtmyer lattice with antithetic points and nshift random shifts; Genz's
# sequential conditioning of the normal vector truncated to z >= t, with the probabilities of
# the conditioning and the exponential factor exp(logE) combined in the log domain. Returns one
# vector of whitened integrals I(nu) per shift.
function Iqmc(IL::MIndex{d}, S::Matrix{Float64}, Lc::Matrix{Float64}, t::Vector{Float64},
              nqmc::Integer, nshift::Integer, seed::Integer, logE::Real) where {d}
    perm = genz_order(S, t)                   # integration order (most restrictive first)
    Sp = S[perm, perm]; tp = t[perm]
    Lp = cholesky(Symmetric(Sp)).L
    primes_ = Int[]; c = 2
    while length(primes_) < d
        all(c % q != 0 for q in primes_) && push!(primes_, c)
        c += 1
    end
    alpha = sqrt.(Float64.(primes_))
    rng = _Lcg(seed)
    Id = Matrix{Float64}(LinearAlgebra.I, d, d)
    out = Vector{Vector{Float64}}()
    for sh in 1:nshift
        shift = [_rand(rng) for _ in 1:d]
        n = 2nqmc
        U = zeros(d, n)
        for k in 1:nqmc, j in 1:d
            u = mod(k*alpha[j] + shift[j], 1.0)
            U[j, k] = u; U[j, nqmc+k] = 1 - u
        end
        W = zeros(d, n); lw = zeros(n)
        Y = zeros(d)
        for k in 1:n
            for j in 1:d
                s = 0.0
                for l in 1:j-1
                    s += Lp[j, l]*Y[l]
                end
                aj = (tp[j] - s)/Lp[j, j]
                lpa = logPhibar(aj)                          # log P(Z >= a_j)
                lw[k] += lpa
                Y[j] = tailinv(aj, lpa + log(U[j, k]))      # P(Z >= Y_j) = U P(Z >= a_j)
            end
            z = Lp*Y
            for j in 1:d
                W[perm[j], k] = z[j] - tp[j]
            end
        end
        push!(out, Isum(IL, Id, LowerTriangular(Lc) \ W, exp.(logE .+ lw) ./ n))
    end
    return out
end

# log P(Z >= a), Z standard normal, without underflow (erfcx for a > 0)
logPhibar(a::Float64) = a > 0 ? log(erfcx(a/sqrt(2))/2) - a^2/2 : log(erfc(a/sqrt(2))/2)

# y >= a with log P(Z >= y) = lq: erfcinv where the tail probability is representable, and
# otherwise Newton's method on log P(Z >= y), started from the exponential approximation
function tailinv(a::Float64, lq::Float64)
    y = lq < -600 ? NaN : sqrt(2)*erfcinv(2*exp(lq))
    if !isfinite(y)
        y = a + (logPhibar(a) - lq)/max(a, 1.0)               # P(Z >= a + e/a) ~ e^{-e} P(Z >= a)
        for it in 1:20
            f = logPhibar(y) - lq
            h = -sqrt(2/pi)/erfcx(y/sqrt(2))                  # d/dy log P(Z >= y) (y > 0)
            y = max(a, y - f/h)
        end
    end
    return max(y, a)
end

# variable prioritization of Genz and Bretz: at each step the remaining variable with the
# smallest conditional probability of exceeding its limit, the earlier variables set to their
# conditional expectations given the truncation
function genz_order(S::Matrix{Float64}, t::Vector{Float64})
    d = length(t)
    perm = collect(1:d); Sp = copy(S); tp = copy(t)
    L = zeros(d, d); y = zeros(d)
    for j in 1:d
        best = j; bestp = Inf
        for i in j:d
            v = Sp[i, i] - sum(abs2, L[i, 1:j-1])
            a = (tp[i] - dot(L[i, 1:j-1], y[1:j-1]))/sqrt(max(v, eps()))
            pa = erfc(a/sqrt(2))/2
            pa < bestp && (best = i; bestp = pa)
        end
        if best != j
            perm[[j, best]] = perm[[best, j]]
            Sp[[j, best], :] = Sp[[best, j], :]; Sp[:, [j, best]] = Sp[:, [best, j]]
            tp[[j, best]] = tp[[best, j]]
            L[[j, best], :] = L[[best, j], :]
        end
        L[j, j] = sqrt(Sp[j, j] - sum(abs2, L[j, 1:j-1]))
        for i in j+1:d
            L[i, j] = (Sp[i, j] - dot(L[i, 1:j-1], L[j, 1:j-1]))/L[j, j]
        end
        a = (tp[j] - dot(L[j, 1:j-1], y[1:j-1]))/L[j, j]
        pa = erfc(a/sqrt(2))/2
        y[j] = pa > 0 ? exp(-a^2/2)/sqrt(2pi)/pa : a
    end
    return perm
end

# small deterministic generator for the random shifts (no dependence on Random)
mutable struct _Lcg
    s::UInt64
end
function _rand(g::_Lcg)
    g.s = g.s*0x5851f42d4c957f2d + 0x14057b7ef767814f
    return (g.s >> 11)*2.0^-53
end

# d = 3, exact: I(nu) = sum_{rho <= nu} binom(nu, rho)(-tau)^(nu-rho) J(rho - 1), with J from
# Lemma 2 (port of the variable-precision program of note 20)
function Iexact3(IL::MIndex{3}, S::Matrix{T}, Si::Matrix{T}, t::Vector{T}, tau::Vector{T}) where {T}
    L = IL.N
    Jm = Jtable3(L, S, Si, t)                 # Jm[r1+2, r2+2, r3+2] = J(r), r_j >= -1
    n = L + 1
    X = Jm[1:n, 1:n, 1:n]                     # X[rho+1] = J(rho - 1)
    for j in 1:3
        Tm = zeros(T, n, n)
        for nn in 0:L, rr in 0:nn
            Tm[nn+1, rr+1] = T(binomial(big(nn), rr))*(-tau[j])^(nn - rr)
        end
        perm = j == 1 ? (1, 2, 3) : (j == 2 ? (2, 1, 3) : (3, 1, 2))
        Y = permutedims(X, perm)
        Y = reshape(Tm*reshape(Y, n, :), n, n, n)
        X = permutedims(Y, invperm(perm))
    end
    return [X[nu[1]+1, nu[2]+1, nu[3]+1] for nu in IL.list]
end

function hermite3_grid(N::Integer, Si::Matrix{T}, x::Vector{T}) where {T}
    I = MIndex(3, N)
    h = hermite_mv(I, Si, reshape(x, 3, 1))[1, :]
    He = zeros(T, N + 1, N + 1, N + 1)
    for (k, nu) in enumerate(I.list)
        He[nu[1]+1, nu[2]+1, nu[3]+1] = h[k]
    end
    return He
end

function Jtable3(L::Integer, S::Matrix{T}, Si::Matrix{T}, t::Vector{T}) where {T}
    Jm = zeros(T, L + 1, L + 1, L + 1)
    He = hermite3_grid(L, Si, t)
    phi3 = exp(-dot(t, Si*t)/2)/((2*T(pi))^(T(3)/2)*sqrt(det(S)))
    Jm[2:L+1, 2:L+1, 2:L+1] = phi3 .* He[1:L, 1:L, 1:L]          # (A) r >= 0
    for i in 1:3                                                     # (B) r_i = -1
        Jx = setdiff(1:3, i)
        SJ = S[Jx, Jx]; SJi = inv(SJ); tJ = t[Jx]
        bJ = SJi*tJ; cJ = (S[i:i, Jx]*SJi)[1, :]
        cv = S[i, i] - dot(S[i, Jx], SJi*S[Jx, i])
        g = exp(-dot(tJ, SJi*tJ)/2)/(2*T(pi)*sqrt(det(SJ)))*erfc((t[i] - dot(cJ, tJ))/sqrt(2cv))/2
        Jr = zeros(T, L, L); Jr[1, 1] = g
        for k in 1:L-2, q1 in k:-1:0
            q = (q1, k - q1)
            jj = q[1] > 0 ? 1 : 2
            r = jj == 1 ? (q[1] - 1, q[2]) : (q[1], q[2] - 1)
            v = bJ[jj]*Jr[r[1]+1, r[2]+1]
            r[1] > 0 && (v -= r[1]*SJi[jj, 1]*Jr[r[1], r[2]+1])
            r[2] > 0 && (v -= r[2]*SJi[jj, 2]*Jr[r[1]+1, r[2]])
            full = zeros(Int, 3); full[Jx] .= r                      # r + e_i: r_i = 0
            v -= cJ[jj]*Jm[full[1]+2, full[2]+2, full[3]+2]
            Jr[q[1]+1, q[2]+1] = v
        end
        for q1 in 0:L-2, q2 in 0:L-2-q1
            idx = zeros(Int, 3); idx[i] = 1; idx[Jx] .= (q1 + 2, q2 + 2)
            Jm[idx...] = Jr[q1+1, q2+1]
        end
    end
    for j in 1:3                                                     # (C) r_I = -1, I = other two
        Ix = setdiff(1:3, j)
        Sjj = S[j, j]
        SI = S[Ix, Ix] - S[Ix, j]*S[j:j, Ix]/Sjj
        g = exp(-t[j]^2/(2Sjj))/sqrt(2*T(pi)*Sjj)*orthant2(t[Ix] - S[Ix, j]*t[j]/Sjj, SI)
        Jr = zeros(T, L); Jr[1] = g
        for q in 0:L-2
            v = t[j]/Sjj*Jr[q+1]
            q > 0 && (v -= q/Sjj*Jr[q])
            for i in Ix
                idx = zeros(Int, 3); idx[Ix] .= 1; idx[i] = 2; idx[j] = q + 2
                v -= S[i, j]/Sjj*Jm[idx...]
            end
            Jr[q+2] = v
        end
        for q in 0:L-1
            idx = zeros(Int, 3); idx[Ix] .= 1; idx[j] = q + 2
            Jm[idx...] = Jr[q+1]
        end
    end
    Jm[1, 1, 1] = orthant3(t, S)                                    # (D) r = (-1,-1,-1)
    return Jm
end

"""
    orthant3(t, S)

``P(Z ≥ t)``, ``Z ∼ N(0,S)`` trivariate, by adaptive quadrature over ``z_1`` of the bivariate
conditional orthant probability ([`orthant2`](@ref)), in the working precision.
"""
function orthant3(t::AbstractVector{T}, S::AbstractMatrix{T}) where {T}
    s1 = sqrt(S[1, 1])
    b = S[2:3, 1] ./ S[1, 1]
    Sc = S[2:3, 2:3] - S[2:3, 1]*S[1:1, 2:3]/S[1, 1]
    rtol = T === Float64 ? 1e-13 : eps(T)^T(0.75)
    f(z) = exp(-z^2/(2S[1, 1]))/(sqrt(2*T(pi))*s1)*orthant2(t[2:3] .- b .* z, Sc; rtol=rtol/10)
    v, _ = quadgk(f, t[1], T(Inf); rtol=rtol, order=15, maxevals=10^5)
    return v
end


"""
    spa_orthant(Kfun, x, order; kwargs...) -> (; Fup, tau, err)

``P(X ≥ x)`` of orders `1:order` in any dimension, with the inclusion-exclusion (17) of the paper
when the saddlepoint has negative components: every term is the orthant probability of a marginal
with some components negated, computed by `spa_uni` (one dimension), `spa_bi` (two), `spa_tri`
(three) or [`spa_mv`](@ref) (four or more), recursively when its own saddlepoint has negative
components. `kwargs` are passed to `spa_tri` and `spa_mv`, and `Kwhite` also to `spa_bi`.
`err[M]` adds the absolute error estimates of the terms of order `M` (quadrature: the change with
half the nodes, ``|F|`` `qerr` for the bivariate terms; simulation: the standard error), so that
the amplification of these errors by cancellation among the terms is visible relative to `Fup`;
it is `NaN` when a term has no estimate, and it does not include the asymptotic error of the
expansions. The keyword `Kwhite` (see `spa_mv`) is transformed for every term: the term with the
kept components and those in ``T`` negated has the cumulant generating function ``K(Bu)``, where
``B`` (``d×k``) has the column ``±e_j`` for each kept ``j``, and is passed the callback
`(nu, u, M) -> Kwhite(nu, u, B*M)`; `Kwhite` must therefore accept a rectangular ``d×k`` matrix
(the cumulant generating function of ``M'X``, of dimension ``k``), as it does when it is written
for ``K(Mu)`` with a general matrix ``M``.
"""
function spa_orthant(Kfun, x::AbstractVector, order::Integer; Kwhite=nothing, kwargs...)
    isempty(x) && throw(ArgumentError("spa_orthant: x must be a nonempty vector"))
    all(isfinite, x) || throw(ArgumentError("spa_orthant: x must be finite"))
    order >= 1 || throw(ArgumentError("spa_orthant: order must be a positive integer"))
    d = length(x)
    T = float(eltype(x)); x = T.(collect(x))
    if d == 1
        K1 = (k, s) -> Kfun((k,), [s])
        return (; Fup=uni_upper(K1, x[1], order), tau=T[NaN], err=zeros(T, order))
    end
    tau, _ = spa_saddle(Kfun, x, zeros(T, d))
    if all(tau .>= 0)
        r = _orthant_core(Kfun, x, order, tau; Kwhite, kwargs...)
        return (; Fup=r.Fup, tau, err=r.err)
    end
    Nn = findall(tau .< 0)
    Fup = zeros(T, order); err = zeros(T, order)
    for mask in 0:2^length(Nn)-1
        Tset = [Nn[i] for i in eachindex(Nn) if (mask >> (i - 1)) & 1 == 1]
        sgn = (-1)^length(Tset)
        keep = setdiff(1:d, setdiff(Nn, Tset))
        flip = ones(Int, d); flip[Tset] .= -1
        if isempty(keep)
            Fup .+= sgn
            continue
        end
        k = length(keep)
        function Km(nu, t)                       # marginal of the kept components, flipped
            nf = zeros(Int, d); tf = zeros(eltype(t), d)
            nf[keep] .= collect(nu); tf[keep] .= t
            return prod(flip[keep] .^ collect(nu))*Kfun(Tuple(nf), flip .* tf)
        end
        xk = (flip .* x)[keep]
        Kw = nothing
        if Kwhite !== nothing
            B = zeros(d, k)
            for (c, j) in enumerate(keep)
                B[j, c] = flip[j]
            end
            Kw = (nu, u, M) -> Kwhite(nu, u, B*M)         # K_B(M u) = K(B M u)
        end
        if k == d
            r = _orthant_core(Km, xk, order, abs.(tau); Kwhite=Kw, kwargs...)
        else
            r = spa_orthant(Km, xk, order; Kwhite=Kw, kwargs...)
        end
        Fup .+= sgn .* r.Fup; err .+= r.err
    end
    return (; Fup, tau, err)
end

function _orthant_core(Kfun, x, order, tau0; Kwhite=nothing, kwargs...)
    d = length(x); T = eltype(x)
    if d == 2
        K2f = (nu, t) -> Kfun(Tuple(nu), t)
        r = spa_bi(K2f, x, order; tau0=tau0, Kwhite)
        return (; Fup=r.Fup, tau=r.tau, err=abs.(r.Fup) .* r.qerr)
    elseif d == 3
        r = spa_tri(Kfun, x, order; tau0=tau0, Kwhite, kwargs...)
        return (; Fup=r.Fup, tau=r.tau, err=r.err)
    else
        r = spa_mv(Kfun, x, order; tau0=tau0, Kwhite, kwargs...)
        return (; Fup=r.Fup, tau=r.tau, err=r.err)
    end
end

"""
    mvcc_Kfun(nu, t, m, lam)

Derivative of order `nu` (a `d`-tuple) at `t` of the joint cumulant generating function of
``Q_j = X_j + X_0``, ``j = 1,…,d``, with independent ``X_i ∼ χ^2_{m_i}(λ_i)``,
`m = [m_1, …, m_d, m_0]`, `lam` likewise: ``K(t) = \\sum_j K_j(t_j) + K_0(\\sum_j t_j)``;
`NaN` outside the domain.
"""
function mvcc_Kfun(nu, t::AbstractVector, m, lam)
    d = length(t); T = float(eltype(t))
    (all(1 .- 2 .* t .> 0) && 1 - 2sum(t) > 0) || return T(NaN)
    k = sum(nu)
    v = ncx2_Kder(sum(t), m[d+1], lam[d+1], k)[end]
    nz = findall(>(0), collect(nu))
    if isempty(nz)
        for j in 1:d
            v += ncx2_Kder(t[j], m[j], lam[j], 0)[1]
        end
    elseif length(nz) == 1
        j = nz[1]
        v += ncx2_Kder(t[j], m[j], lam[j], k)[end]
    end
    return v
end

"""
    mvcc_exact(x, m, lam) -> (; Fup, pdf)

``P(Q ≥ x)`` and the joint density of ``Q_j = X_j + X_0`` (see [`mvcc_Kfun`](@ref)) by
one-dimensional adaptive integration over ``X_0`` (relative tolerance 1e-12).
"""
function mvcc_exact(x::AbstractVector, m, lam)
    d = length(x)
    f0(s) = ncx2_exact(s, m[d+1], lam[d+1]).pdf
    up(s, j) = s >= x[j] ? 1.0 : ncx2_exact(x[j] - s, m[j], lam[j]).upp
    dn(s, j) = s >= x[j] ? 0.0 : ncx2_exact(x[j] - s, m[j], lam[j]).pdf
    br = sort(unique(Float64.(x)))
    Fup = quadgk(s -> f0(s)*prod(up(s, j) for j in 1:d), 0, br..., Inf; rtol=1e-12)[1]
    pdf = quadgk(s -> f0(s)*prod(dn(s, j) for j in 1:d), 0, br[1]; rtol=1e-12)[1]
    return (; Fup, pdf)
end
