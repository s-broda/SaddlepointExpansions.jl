# Studentized means: the density and distribution function of Q = X/sqrt(Y), Y > 0, from the
# joint cumulant generating function K(s, t) of (X, Y). The expansions of Bao, Broda and Kan,
# "Inversion formulas and saddlepoint expansions for Studentized means": the density by the
# bivariate amplitude expansion of Theorem 2, the distribution function by the Hauschildt-Robinson
# type expansion of Theorem 3 after the reduction to one dimension, both built from one Taylor
# polynomial of K at the saddlepoint (shat, that). The saddlepoint equations are
#     K_s = q v,  K_t = v^2,  q s + 2 t v = 0,
# and the saddlepoints for all q lie on the locus s K_s + 2 t K_t = 0, on which v = K_t^{1/2} and
# q = K_s/v, which is what the quantile search uses.

_stud_check(K::CGF, name) = K.d == 2 ||
    throw(ArgumentError("$name: K must be the joint cumulant generating function of (X, Y), of dimension 2"))

# the Taylor polynomial of K at (s, t) of degree N; an error outside the domain
function _stud_jet(K::CGF, s::T, t::T, N::Integer) where {T}
    P = _taylor(K, [s, t], N)
    P === nothing && error("studentized: the point (s, t) = ($(Float64(s)), $(Float64(t))) is outside the domain of K")
    return P
end

# Newton's method on the saddlepoint equations from (s, t, v), with the Jacobian from the
# second-order Taylor polynomial of K
function _stud_newton3(K::CGF, q::T, s::T, t::T, v::T; tol=50eps(T), maxit=60) where {T}
    for _ in 1:maxit
        P = _taylor(K, [s, t], 2)
        P === nothing && error("outside the domain")
        Ks, Kt = _coef(T, P, (1, 0)), _coef(T, P, (0, 1))
        Kss, Kst, Ktt = _coef(T, P, (2, 0)), _coef(T, P, (1, 1)), _coef(T, P, (0, 2))
        F = [Ks - q*v, Kt - v*v, q*s + 2*t*v]
        A = [Kss Kst -q; Kst Ktt -2v; q 2v 2t]
        d = A \ F
        s -= d[1]; t -= d[2]; v -= d[3]
        sum(abs, d) < tol*(1 + abs(s) + abs(t) + abs(v)) && return s, t, v
    end
    error("the saddlepoint iteration did not converge at q = $(Float64(q))")
end

# continuation from qa (solution s, t, v) to qb, halving the step on failure
function _stud_continue(K::CGF, qa, qb, s, t, v, depth=0)
    try
        return _stud_newton3(K, qb, s, t, v)
    catch err
        err isa ArgumentError && rethrow()
        depth >= 12 && rethrow()
        qm = (qa + qb)/2
        s, t, v = _stud_continue(K, qa, qm, s, t, v, depth + 1)
        return _stud_continue(K, qm, qb, s, t, v, depth + 1)
    end
end

"""
    studentized_saddlepoint(K::CGF, q) -> (; s, t, v, q0)

The saddlepoint ``(\\hat s, \\hat t, \\hat v)`` of the expansions for ``Q = X/\\sqrt{Y}`` at `q`,
the root of ``K_s = q v``, ``K_t = v^2``, ``q s + 2 t v = 0`` with ``\\hat v > 0``, where `K` is the
joint cumulant generating function of ``(X, Y)``, ``Y > 0``. It is found by Newton's method with
continuation in `q` from ``q_0 = E(X)/\\{E(Y)\\}^{1/2}``, where ``\\hat s = \\hat t = 0``.
"""
function studentized_saddlepoint(K::CGF, q::Real)
    _stud_check(K, "studentized_saddlepoint")
    T = float(typeof(q)); q = T(q)
    P0 = _stud_jet(K, zero(T), zero(T), 2)
    EU, EV = _coef(T, P0, (1, 0)), _coef(T, P0, (0, 1))
    EV > 0 || error("studentized_saddlepoint: E(Y) must be positive")
    q0 = EU/sqrt(EV)
    s, t, v = zero(T), zero(T), sqrt(EV)
    steps = max(4, Int(floor(abs(q - q0)/T(1//4))) + 1)
    for k in 1:steps
        s, t, v = _stud_continue(K, q0 + (q - q0)*(k - 1)/steps, q0 + (q - q0)*k/steps, s, t, v)
    end
    return (; s, t, v, q0)
end

"""
    studentized_locus(K::CGF, s, tguess) -> (; t, v, q)

The point of the saddlepoint locus ``s K_s + 2 t K_t = 0`` with abscissa `s`, by Newton's method in
``t`` from `tguess`, with ``v = K_t^{1/2}`` and the value ``q = K_s/v`` whose saddlepoint it is.
Eliminating ``q`` and ``v`` from the saddlepoint equations shows that the saddlepoints for all
``q`` lie on this curve through the origin, so that a value of ``\\hat s`` determines the
saddlepoint and ``q`` without solving the three equations; [`studentized_quantile`](@ref) searches
along it.
"""
function studentized_locus(K::CGF, s::Real, tguess::Real; tol=nothing, maxit=60)
    _stud_check(K, "studentized_locus")
    T = float(promote_type(typeof(s), typeof(tguess))); s = T(s); t = T(tguess)
    tol = tol === nothing ? 50eps(T) : T(tol)
    for _ in 1:maxit
        P = _stud_jet(K, s, t, 2)
        Ks, Kt = _coef(T, P, (1, 0)), _coef(T, P, (0, 1))
        Kst, Ktt = _coef(T, P, (1, 1)), _coef(T, P, (0, 2))
        g = s*Ks + 2*t*Kt; gp = s*Kst + 2*Kt + 2*t*Ktt
        t -= g/gp
        if abs(g/gp) < tol*(1 + abs(t))
            P = _stud_jet(K, s, t, 2)
            Ks, Kt = _coef(T, P, (1, 0)), _coef(T, P, (0, 1))
            Kt > 0 || error("studentized_locus: K_t is not positive at the point found")
            v = sqrt(Kt)
            return (; t, v, q=Ks/v)
        end
    end
    error("studentized_locus: the iteration did not converge at s = $(Float64(s))")
end

# ----- the density: Theorem 2, in whitened coordinates with a complex factorization of psi''

# a complex L with psi'' = L L' from the eigen-decomposition H = V diag(lambda) V' of the 2 x 2
# symmetric matrix, L = V diag(sqrt(lambda)) (the Hessian is indefinite when that < 0, and its
# (1, 1) entry vanishes in symmetric cases, so a triangular factorization is not used)
function _complex_chol(H::AbstractMatrix{T}) where {T}
    C = Complex{T}
    a, b, c = H[1, 1], H[2, 1], H[2, 2]
    th = atan(2b, a - c)/2
    cs, sn = cos(th), sin(th)
    V = [cs -sn; sn cs]
    l1 = a*cs*cs + 2b*sn*cs + c*sn*sn; l2 = a*sn*sn - 2b*sn*cs + c*cs*cs
    (l1 != 0 && l2 != 0) || error("studentized_density: psi'' is singular at the saddlepoint")
    r1, r2 = sqrt(C(l1)), sqrt(C(l2))
    L = V*[r1 zero(C); zero(C) r2]
    Lit = V*[1/r1 zero(C); zero(C) 1/r2]             # inv(L)' = V diag(1/sqrt(lambda))
    return L, Lit
end

_cpow(x, r) = exp(r*log(x))

"""
    studentized_density(K::CGF, q; order=3, saddle=nothing) -> (; f, saddle, status, info)

Saddlepoint expansions of orders `1:order` of the density at `q` of ``Q = X/\\sqrt{Y}``, ``Y > 0``,
from the joint cumulant generating function `K` of ``(X, Y)`` (Theorem 2 of Bao, Broda and Kan):
`f[M]` keeps the terms ``m ≤ M - 1`` of the expansion in powers of ``n^{-1}`` and has relative
error ``O(n^{-M})`` (the "``n``" is absorbed in `K`, see [`sample_mean`](@ref)). With
``\\psi(s, t) = K(s, t) + q^2s^2/(4t)`` and the saddlepoint ``(\\hat s, \\hat t, \\hat v)`` of
[`studentized_saddlepoint`](@ref),
```math
f_Q(q) ≈ \\frac{\\hat v^2 e^{\\psi(\\hat s, \\hat t)}}{\\sqrt{\\pi}\\,\\{\\hat t\\,|\\psi''(\\hat s, \\hat t)|\\}^{1/2}}
\\Big\\{1 + \\sum_{m ≥ 1} b_m(h_0) + b_{m-1}(h_1)\\Big\\},
```
the amplitude expansion of a bivariate integral with amplitude
``h_0 + h_1 = (s/\\hat s)^2(t/\\hat t)^{-5/2} + (t/\\hat t)^{-3/2}/(2\\hat t\\hat v^2)``; the
Hessian ``\\psi''`` is indefinite where ``\\hat t < 0``, and the Hermite moments are evaluated in
the coordinates of a complex factorization ``\\psi'' = LL'``. The terms of high order cancel by
many orders of magnitude when ``|\\hat t|`` is large; `info.kappa` is the cancellation indicator,
and with `refine = true` (the default) a `Float64` computation whose indicator exceeds ``10^6``
is repeated in `BigFloat` with a precision chosen from the indicator (`info.refined` is then the
number of bits). `saddle` may supply the saddlepoint (`(; s, t, v)`).
"""
function studentized_density(K::CGF, q::Real; order::Integer=3, saddle=nothing, refine::Bool=true)
    _stud_check(K, "studentized_density"); _checkorder("studentized_density", order)
    T = float(typeof(q)); q = T(q); M = order
    r = _studentized_density(K, q, M, saddle)
    if refine && T === Float64 && r.info.kappa[end] > 1e6
        # recompute in BigFloat with the precision the cancellation calls for
        prec = 64 + 4*ceil(Int, log2(r.info.kappa[end]))
        rb = setprecision(BigFloat, prec) do
            _studentized_density(K, big(q), M, saddle === nothing ? nothing : (; s=big(saddle.s), t=big(saddle.t), v=big(saddle.v)))
        end
        return (; f=Float64.(rb.f), saddle=(; s=Float64(rb.saddle.s), t=Float64(rb.saddle.t), v=Float64(rb.saddle.v)),
                status=rb.status == "ok" ? "ok" : "refinement failed", info=(; rb.info..., refined=prec))
    end
    return r
end

function _studentized_density(K::CGF, q::T, M::Integer, saddle) where {T}
    sd = saddle === nothing ? studentized_saddlepoint(K, q) : saddle
    s, t, v = T(sd.s), T(sd.t), T(sd.v)
    t != 0 || error("studentized_density: the saddlepoint has t = 0 (q is the point q0 = E(X)/E(Y)^{1/2}); the expansion is not evaluated there")
    D = max(6(M - 1), 2)
    P = _stud_jet(K, s, t, D)
    sp = _space(2, D)
    vv = variables(T, sp)
    psi = P + (s + vv[1])^2*(q*q/4)/(t + vv[2])
    psi0 = constant_term(psi)
    H = [_coef(T, psi, (2, 0)) _coef(T, psi, (1, 1)); _coef(T, psi, (1, 1)) _coef(T, psi, (0, 2))]
    detH = H[1, 1]*H[2, 2] - H[1, 2]^2
    t*detH > 0 || error("studentized_density: that*det(psi'') is not positive at the saddlepoint")
    pref = v^2*exp(psi0)/sqrt(T(pi)*t*detH)
    C = Complex{T}
    L, Lit = _complex_chol(H)
    u = variables(C, sp)
    w = Lit*u                                             # v = L^{-T} u: whitened coordinates
    psiu = evaluate(psi, w)
    # Theta: the terms of degree 3..2M of psi in the whitened coordinates
    zd = (0, 0)
    theta = Dict{NTuple{2,Int},C}()
    for nu in mi_list(2, min(2M, D))
        sum(nu) >= 3 || continue
        theta[nu] = C(getcoeff(psiu, nu))
    end
    Theta = taylor_from_coefficients(2, D, theta; T=C)
    # the amplitudes, as polynomials in u of degree <= 2(M-1)
    a0 = _truncate(((s + w[1])/s)^2*_cpow((t + w[2])/t, -T(5)/2), 2(M - 1), 2, D, C)
    a1 = _truncate(_cpow((t + w[2])/t, -T(3)/2)/(2*t*v*v), 2(M - 1), 2, D, C)
    he(nu) = prod(iseven(n) ? (-1)^(n ÷ 2)*_dfact(n - 1) : 0 for n in nu; init=1)
    R = 2(M - 1)
    Pr = Vector{TaylorN{C}}(undef, R + 1)
    Pr[1] = taylor_from_coefficients(2, D, Dict(zd => one(C)); T=C)
    for r in 1:R
        Pr[r+1] = Pr[r]*Theta/r
    end
    exps = mi_list(2, D)
    # b_m(a) = sum_{r=0}^{2m} sum_{|nu| = 2m+2r} [Theta^r/r! a]_nu He_nu(0)
    function bcoef(m, a)
        tot = zero(C); abssum = zero(T)
        for r in 0:2m
            Q = Pr[r+1]*a
            for nu in exps
                sum(nu) == 2m + 2r || continue
                term = getcoeff(Q, nu)*he(nu)
                tot += term; abssum += abs(term)
            end
        end
        return tot, abssum
    end
    f = zeros(T, M); kappa = zeros(T, M)
    br = one(T); ab = one(T)
    f[1] = pref; kappa[1] = one(T)
    for m in 1:M-1
        b0, s0 = bcoef(m, a0); b1, s1 = bcoef(m - 1, a1)
        br += real(b0 + b1); ab += s0 + s1
        f[m+1] = pref*br; kappa[m+1] = ab/abs(br)
    end
    status = kappa[end]*eps(T) > 1e-10 ? "coefficients inaccurate" : "ok"
    return (; f, saddle=(; s, t, v), status, info=(; pref, psi0, H, kappa))
end

# ----- the distribution function: Theorem 3 after the reduction to one dimension

# the univariate series in ds of the coefficient of dt^j of the bivariate Taylor polynomial J
function _stud_reduction(K::CGF, q::T, s0::T, t0::T, p::Integer) where {T}
    jmax = p ÷ 2
    N = max(p + 2, 2jmax + 2 + p + 2)
    J = _stud_jet(K, s0, t0, N)
    sp = _space(2, N)
    vv = variables(T, sp)
    psi = J + (s0 + vv[1])^2*(q*q/4)/(t0 + vv[2])
    psi_t = differentiate(psi, 2); psi_tt = differentiate(psi_t, 2)
    ds1 = Taylor1(T, N)
    compose(F::TaylorN, tau::Taylor1) = evaluate(F, [ds1, tau])
    tau = zero(ds1)                                        # t(s) - that, nilpotent in ds
    for _ in 1:N+1
        g = compose(psi_t, tau); gt = compose(psi_tt, tau)
        tau = tau - g/gt
        tau[0] = zero(T)
    end
    Psi = compose(psi, tau)
    su = s0 + ds1; tu = t0 + tau
    v = -su*q/(2tu)
    Ktt = compose(differentiate(differentiate(J, 2), 2), tau)
    arg = 2tu*Ktt + 4v*v
    constant_term(arg) > 0 || error("studentized_tailprob: 2 t K_tt + 4 v^2 is not positive at the saddlepoint")
    H0 = 2v/sqrt(arg)
    C = Complex{T}
    cplx(a::Taylor1{T}) = Taylor1(C.(a.coeffs))
    p2 = compose(psi_tt, tau)
    sq = sqrt(cplx(p2))                                    # complex where psi_tt < 0
    dpsi = Dict(k => cplx(compose(_dt(psi, k), tau)) for k in 3:2jmax+2)
    theta = Dict(k => dpsi[k]/(sq^k*factT(T, k)) for k in 3:2jmax+2)
    gam(l) = cplx((one(tu)/tu)^l)/sq^l*_gbinomial(-T(3)/2, l)
    P2 = 2jmax
    zero1 = zero(cplx(tu))
    d = [zero1 for _ in 0:P2, _ in 0:P2]
    d[1, 1] = one(zero1)
    for r in 1:P2, j in r:P2
        acc = zero1
        for k in 1:j-r+1
            acc += theta[k+2]*d[r, j-k+1]
        end
        d[r+1, j+1] = acc/r
    end
    hh = hcoef(T, 2P2 + 2)
    a_ij(i, j) = sum(d[k+1, j+1]*hh[i+k+1] for k in 0:j; init=zero1)
    beta = [sum(gam(2j - k)*a_ij(j, k) for k in 0:2j; init=zero1) for j in 1:jmax]
    Psi_c = [Psi[k] for k in 0:p+2]
    Hk_c = Vector{Vector{T}}()
    push!(Hk_c, [H0[l] for l in 0:p])
    for k in 1:jmax
        Hk = cplx(H0)*beta[k]
        for l in 0:p-2k
            abs(imag(Hk[l])) <= sqrt(eps(T))*(1 + abs(Hk[l])) ||
                error("studentized_tailprob: a coefficient of H_$k is not real ($(Hk[l])); the expansion does not apply at this q")
        end
        push!(Hk_c, [real(Hk[l]) for l in 0:p-2k])
    end
    return Psi_c, Hk_c
end

function _dt(J::TaylorN, k::Integer)
    for _ in 1:k
        J = differentiate(J, 2)
    end
    return J
end

_gbinomial(a, k::Integer) = prod((a - i)/(i + 1) for i in 0:k-1; init=one(a))

# I_m(u), m = 0..mmax, u >= 0: I_0 = Phi(-u), I_{m+1} = He_m(0) phi(u) - u I_m (forward recursion),
# replaced for Float64 and u > 2.5 by the Gauss-Hermite representation of spa_uni
function _stud_Itable(u::T, mmax::Integer) where {T}
    phi = normpdf(u)
    if T === Float64 && u > 2.5
        return phi .* Iherm(u, mmax)
    end
    h = hcoef(T, mmax ÷ 2 + 1)
    I = Vector{T}(undef, mmax + 1)
    I[1] = normcdf(-u)
    for m in 0:mmax-1
        I[m+2] = (iseven(m) ? h[m ÷ 2 + 1] : zero(T))*phi - u*I[m+1]
    end
    return I
end

# the Hauschildt-Robinson type sum (Theorem 3) from the Taylor coefficients Psi_c[k+1] of Psi and
# Hk_c[k+1][l+1] of H_k at shat: approximations to P(Q > q) with the terms up to n^{-j/2}, j = 0..p,
# with the reflection for shat < 0
function _stud_hrsum(Psi_c::Vector{T}, Hk_c::Vector{Vector{T}}, s0::T, p::Integer) where {T}
    sg = s0 >= 0 ? 1 : -1
    Psi2 = 2*Psi_c[3]
    Psi2 > 0 || error("studentized_tailprob: Psi''(shat) is not positive")
    theta = Dict(k => sg^k*Psi_c[k+1]/Psi2^(T(k)/2) for k in 3:p+2)
    gam = [[sg^l*Hk_c[k+1][l+1]/Psi2^(T(l)/2) for l in 0:length(Hk_c[k+1])-1] for k in 0:length(Hk_c)-1]
    d = Dict{Tuple{Int,Int},T}((0, 0) => one(T))
    for r in 1:p, j in r:p
        d[(r, j)] = sum(theta[k+2]*get(d, (r - 1, j - k), zero(T)) for k in 1:j-r+1; init=zero(T))/r
    end
    w = sqrt(-2*Psi_c[1]); u = abs(s0)*sqrt(Psi2)
    En = exp((u^2 - w^2)/2)
    I = _stud_Itable(u, 3p + 1)
    out = zeros(T, p + 1); kappa = zeros(T, p + 1)
    tot = zero(T); abssum = zero(T)
    for j in 0:p
        for k in 0:j÷2, m in 0:j-2k
            l = j - 2k - m
            for r in 0:m
                drm = (r == 0 && m == 0) ? one(T) : (r == 0 ? zero(T) : get(d, (r, m), zero(T)))
                drm == 0 && continue
                term = drm*gam[k+1][l+1]*I[m+l+2r+1]
                tot += term; abssum += abs(term)
            end
        end
        val = En*tot
        out[j+1] = sg == 1 ? val : 1 - val
        kappa[j+1] = abssum/abs(tot)
    end
    return out, kappa
end

"""
    studentized_tailprob(K::CGF, q; order=3, lower=false, saddle=nothing) -> (; F, saddle, status, info)

Saddlepoint expansions of orders `1:order` of ``P(Q ≥ q)`` for ``Q = X/\\sqrt{Y}``, ``Y > 0``,
from the joint cumulant generating function `K` of ``(X, Y)`` (Theorem 3 of Bao, Broda and Kan):
`F[M]` keeps the terms of orders ``n^{-m/2}``, ``m ≤ 2M - 1``, and has relative error ``O(n^{-M})``;
`info.Fp[j+1]`, ``j = 0, …, 2M - 1``, are the expansions with the terms up to ``n^{-j/2}``.
`lower = true` gives ``P(Q ≤ q)``. The expansion is of Hauschildt-Robinson type in the variable
``s`` after the ``t``-integral is expanded by Laplace's method: with ``\\Psi(s) = \\psi(s, t(s))``,
``\\psi_t(s, t(s)) = 0``, the leading term is ``E_n\\{H_0(\\hat s)\\bar\\Phi(\\hat u) + …\\}``, and all
Taylor coefficients of ``\\Psi`` and of the amplitudes ``H_k`` at ``\\hat s`` are obtained from the
Taylor polynomial of ``K`` by Newton's method and composition on power series, exact to rounding.
`saddle` may supply the saddlepoint (`(; s, t, v)`); `info.kappa` is the cancellation indicator.
"""
function studentized_tailprob(K::CGF, q::Real; order::Integer=3, lower::Bool=false, saddle=nothing)
    _stud_check(K, "studentized_tailprob"); _checkorder("studentized_tailprob", order)
    T = float(typeof(q)); q = T(q); M = order
    sd = saddle === nothing ? studentized_saddlepoint(K, q) : saddle
    s, t, v = T(sd.s), T(sd.t), T(sd.v)
    t != 0 || error("studentized_tailprob: the saddlepoint has t = 0 (q is the point q0 = E(X)/E(Y)^{1/2}); the expansion is not evaluated there")
    p = 2M - 1
    Psi_c, Hk_c = _stud_reduction(K, q, s, t, p)
    Fp, kappa = _stud_hrsum(Psi_c, Hk_c, s, p)
    lower && (Fp = 1 .- Fp)
    F = [Fp[2m] for m in 1:M]
    status = kappa[end] > 1e8 ? "coefficients inaccurate" : "ok"
    return (; F, saddle=(; s, t, v), status, info=(; Fp, kappa, Psi_c, Hk_c))
end

"""
    studentized_quantile(K::CGF, prob; order=3, tol=1e-10) -> (; q, saddle)

The quantile of the saddlepoint distribution function of order `order` ([`studentized_tailprob`](@ref)
with `lower = true`): ``P(Q ≤ q) = `` `prob`. The search runs over ``\\hat s`` along the
saddlepoint locus ([`studentized_locus`](@ref)), where each trial point gives its ``q`` and its
distribution function directly, so that no saddlepoint equation is solved: a bracket in
``\\hat s`` from the origin, then the Illinois iteration.
"""
function studentized_quantile(K::CGF, prob::Real; order::Integer=3, tol::Real=1e-10)
    _stud_check(K, "studentized_quantile"); _checkorder("studentized_quantile", order)
    0 < prob < 1 || throw(ArgumentError("studentized_quantile: prob must be in (0, 1)"))
    T = float(typeof(prob)); prob = T(prob)
    P0 = _stud_jet(K, zero(T), zero(T), 2)
    EU, EV = _coef(T, P0, (1, 0)), _coef(T, P0, (0, 1))
    slope = -EU/(2EV)                                      # t'(0) along the locus
    function at(s, tg)
        r = studentized_locus(K, s, tg)
        Fv = studentized_tailprob(K, r.q; order, lower=true, saddle=(; s, t=r.t, v=r.v)).F[end]
        return Fv, r.q, r.t
    end
    step = T(1//2); s = T(1//10); t = slope*s
    F, q, t = at(s, t)
    dir = F < prob ? 1 : -1
    if dir == -1
        s = -s; t = slope*s; F, q, t = at(s, t)
    end
    sa, Fa, ta = s, F, t
    while (dir == 1 && F < prob) || (dir == -1 && F > prob)
        sa, Fa, ta = s, F, t
        s += dir*step; F, q, t = at(s, t)
        abs(s) > 1e4 && error("studentized_quantile: no bracket found")
    end
    sb, Fb, tb = s, F, t
    fa, fb = Fa - prob, Fb - prob; side = 0
    for _ in 1:200
        sm = (sa*fb - sb*fa)/(fb - fa)
        tg = ta + (tb - ta)*(sm - sa)/(sb - sa)
        Fm, qm, tm = at(sm, tg)
        fm = Fm - prob
        if abs(fm) < tol
            r = studentized_locus(K, sm, tm)
            return (; q=qm, saddle=(; s=sm, t=r.t, v=r.v))
        end
        if sign(fm) == sign(fb)
            sb, fb, tb = sm, fm, tm
            side == 1 && (fa /= 2); side = 1
        else
            sa, fa, ta = sm, fm, tm
            side == -1 && (fb /= 2); side = -1
        end
    end
    error("studentized_quantile: the iteration did not converge")
end
