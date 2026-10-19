# Noncentral chi-squared, singly and doubly noncentral F, and pairs of quadratic forms with a
# common component (Section 6.4 of the paper and Section S2 of its Supplementary Material).

"""
    ncx2_Kder(u, m, lam, N) -> Vector

Derivatives ``K^{(k)}(u)``, `k = 0..N`, of the cumulant generating function of
``χ^2_m(λ)``: ``K(u) = -(m/2)\\log(1-2u) + λu/(1-2u)``.
"""
function ncx2_Kder(u::Real, m::Real, lam::Real, N::Integer)
    T = float(promote_type(typeof(u), typeof(m), typeof(lam)))
    u = T(u); m = T(m); lam = T(lam)
    Kd = zeros(T, N + 1)
    Kd[1] = -m/2*log1p(-2u) + lam*u/(1 - 2u)
    for k in 1:N
        Kd[k+1] = T(2)^(k-1)*factT(T, k - 1)*m/(1 - 2u)^k + T(2)^(k-1)*factT(T, k)*lam/(1 - 2u)^(k+1)
    end
    return Kd
end

"""
    ncx2_spa(x, nu, lam, order) -> (; cdf, pdf, upp, kappa)

Saddlepoint approximations of orders `1:order` of the distribution function ``P(X ≤ x)``, the
density, and the upper tail ``P(X > x)`` (computed directly) of ``χ^2_ν(λ)`` at ``x>0``. The
saddlepoint solves a quadratic in ``1/(1-2s)``, written in the form that is stable for small λ.
"""
function ncx2_spa(x::Real, nu::Real, lam::Real, order::Integer)
    (x > 0 && isfinite(x)) || throw(ArgumentError("ncx2_spa: x must be positive"))
    (nu > 0 && lam >= 0) || throw(ArgumentError("ncx2_spa: invalid nu or lam"))
    order >= 1 || throw(ArgumentError("ncx2_spa: order must be a positive integer"))
    T = float(promote_type(typeof(x), typeof(nu), typeof(lam)))
    x = T(x); nu = T(nu); lam = T(lam)
    y = 2x/(nu + sqrt(nu^2 + 4lam*x))
    s = (1 - 1/y)/2
    Kd = ncx2_Kder(s, nu, lam, 2order + 1)
    r = spa_uni(Kd, s, x, order)
    return (; cdf=r.cdf, pdf=r.pdf, upp=r.upp, kappa=r.kappa)
end

"""
    ncf_spa(f, m, n, lam, order) -> (; cdf, pdf, upp, kappa)

Approximations of orders `1:order` of the distribution function, density and upper tail of the
singly noncentral F, ``F=(Q_1/m)/(Q_2/n)``, ``Q_1∼χ^2_m(λ)``, ``Q_2∼χ^2_n``, at ``f>0``:
``P(F ≤ f)=P(W ≤ 0)``, ``W=Q_1-cQ_2``, ``c=mf/n``, by the Hauschildt-Robinson expansion; the
density by the Rice expansion of the density of ``Q_1/Q_2`` at ``c``, times ``m/n``. The
saddlepoint is the root in ``(-1/(2c), 1/2)`` of a quadratic.
"""
function ncf_spa(f::Real, m::Real, n::Real, lam::Real, order::Integer)
    (f > 0 && isfinite(f)) || throw(ArgumentError("ncf_spa: f must be positive"))
    (m > 0 && n > 0) || throw(ArgumentError("ncf_spa: the degrees of freedom must be positive"))
    (lam >= 0 && isfinite(lam)) || throw(ArgumentError("ncf_spa: lam must be nonnegative"))
    order >= 1 || throw(ArgumentError("ncf_spa: order must be a positive integer"))
    T = float(promote_type(typeof(f), typeof(m), typeof(n), typeof(lam)))
    f = T(f); m = T(m); n = T(n); lam = T(lam)
    c = m*f/n
    A = -4c*(m + n); B = 2(c*m - m + c*lam + 2c*n); C = m + lam - c*n
    disc = sqrt(B^2 - 4A*C)
    q = -(B + copysign(disc, B))/2                 # stable roots q/A and C/q
    shat = T(NaN)
    for r in (q/A, C/q)
        if r > -1/(2c) && r < T(1)/2
            shat = r; break
        end
    end
    N = 2order + 1                                 # derivatives needed by spa_uni
    Kd = ncx2_Kder(shat, m, lam, N) .+ ((-c) .^ (0:N)) .* ncx2_Kder(-c*shat, n, 0, N)
    r = spa_uni(Kd, shat, 0, order)
    K2d = ncx2_Kder(-c*shat, n, 0, 2order)          # g(s) = K_{Q2}'(-c s)
    gd = ((-c) .^ (0:2order-1)) .* K2d[2:2order+1]
    pdf = m/n .* spa_ratio_pdf(Kd, gd, shat, order)
    return (; cdf=r.cdf, pdf, upp=r.upp, kappa=r.kappa)
end

"""
    dncf_saddle(c, m, n, lam1, lam2)

Saddlepoint for the doubly noncentral F: the root in ``(-1/(2c), 1/2)`` of
``K_W'(s) = m/(1-2s) + λ_1/(1-2s)^2 - cn/(1+2cs) - cλ_2/(1+2cs)^2``, ``W=Q_1-cQ_2``. Starting
value: the middle root of the cubic obtained by clearing denominators (trigonometric formula),
refined by safeguarded Newton iteration (bisection on the bracket when a step leaves it).
"""
function dncf_saddle(c::Real, m::Real, n::Real, lam1::Real, lam2::Real)
    T = float(promote_type(typeof(c), typeof(m), typeof(n), typeof(lam1), typeof(lam2)))
    c = T(c); m = T(m); n = T(n); lam1 = T(lam1); lam2 = T(lam2)
    lo = -1/(2c); hi = T(1)/2
    a = 8c^2*(m + n)
    a2 = -4c*(c*lam1 + c*m + 2c*n - lam2 - 2m - n)/a
    a1 = 2(c^2*n - 2c*(lam1 + lam2 + m + n) + m)/a
    a0 = -(m + lam1 - c*(n + lam2))/a
    P = a1 - a2^2/3; Q = 2a2^3/27 - a1*a2/3 + a0
    s = T(NaN)
    if P < 0
        arg = clamp(3Q/(2P)*sqrt(-3/P), -one(T), one(T))
        t = sort([2sqrt(-P/3)*cos(acos(arg)/3 - 2T(pi)*k/3) - a2/3 for k in 0:2])
        s = t[2]
    end
    if !(s > lo && s < hi)
        s = (m + lam1 - c*(n + lam2))/(2(m + lam1)*(1 + c) + 2c*(n + lam2))
        s = max(lo + (hi - lo)/4, min(hi - (hi - lo)/4, s))
    end
    g(s) = m/(1 - 2s) + lam1/(1 - 2s)^2 - c*n/(1 + 2c*s) - c*lam2/(1 + 2c*s)^2
    dg(s) = 2m/(1 - 2s)^2 + 4lam1/(1 - 2s)^3 + 2c^2*n/(1 + 2c*s)^2 + 4c^2*lam2/(1 + 2c*s)^3
    for it in 1:200
        gs = g(s)
        gs < 0 ? (lo = s) : (hi = s)
        sn = s - gs/dg(s)
        (sn > lo && sn < hi) || (sn = (lo + hi)/2)
        if abs(sn - s) <= 4eps(T)*max(abs(s), abs(sn)) || sn == s
            s = sn; break
        end
        s = sn
    end
    return s
end

"""
    dncf_spa(f, m, n, lam1, lam2, order) -> (; cdf, pdf, upp, kappa)

Approximations of orders `1:order` of the distribution function, density and upper tail
(computed directly) of the doubly noncentral F, ``F=(Q_1/m)/(Q_2/n)``, ``Q_1∼χ^2_m(λ_1)``,
``Q_2∼χ^2_n(λ_2)``; `kappa`: cancellation indicator of the distribution function sum.
"""
function dncf_spa(f::Real, m::Real, n::Real, lam1::Real, lam2::Real, order::Integer)
    (f > 0 && isfinite(f)) || throw(ArgumentError("dncf_spa: f must be positive"))
    order >= 1 || throw(ArgumentError("dncf_spa: order must be a positive integer"))
    (lam1 >= 0 && lam2 >= 0 && m > 0 && n > 0) ||
        throw(ArgumentError("dncf_spa: invalid degrees of freedom or noncentralities"))
    T = float(promote_type(typeof(f), typeof(m), typeof(n), typeof(lam1), typeof(lam2)))
    f = T(f); m = T(m); n = T(n); lam1 = T(lam1); lam2 = T(lam2)
    c = m*f/n
    N = 2order + 1                                 # derivatives needed by spa_uni
    shat = dncf_saddle(c, m, n, lam1, lam2)
    Kd = ncx2_Kder(shat, m, lam1, N) .+ ((-c) .^ (0:N)) .* ncx2_Kder(-c*shat, n, lam2, N)
    r = spa_uni(Kd, shat, 0, order)
    K2d = ncx2_Kder(-c*shat, n, lam2, 2order)
    gd = ((-c) .^ (0:2order-1)) .* K2d[2:2order+1]
    pdf = m/n .* spa_ratio_pdf(Kd, gd, shat, order)
    return (; cdf=r.cdf, pdf, upp=r.upp, kappa=r.kappa)
end

"""
    bqfcc_Kfun(nu, t, m, lam)

Derivative of order `nu = (nu1, nu2)` at `t` of the joint cumulant generating function of
``Q_1=X_1+X_0``, ``Q_2=X_2+X_0``, ``X_i∼χ^2_{m_i}(λ_i)`` independent, `m = [m1, m2, m0]`,
`lam = [lam1, lam2, lam0]`: ``K(t_1,t_2)=K_1(t_1)+K_2(t_2)+K_0(t_1+t_2)``; `NaN` outside the
domain.
"""
function bqfcc_Kfun(nu, t::AbstractVector, m, lam)
    T = float(eltype(t))
    (1 - 2t[1] <= 0 || 1 - 2t[2] <= 0 || 1 - 2t[1] - 2t[2] <= 0) && return T(NaN)
    v = ncx2_Kder(t[1] + t[2], m[3], lam[3], nu[1] + nu[2])[end]
    nu[2] == 0 && (v += ncx2_Kder(t[1], m[1], lam[1], nu[1])[end])
    nu[1] == 0 && (v += ncx2_Kder(t[2], m[2], lam[2], nu[2])[end])
    return v
end
