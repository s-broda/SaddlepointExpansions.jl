# Quantiles of the saddlepoint distribution functions and normalized density expansions, both
# computed in saddlepoint space: the point is parametrized by its saddlepoint, x = K'(s), so that
# every trial point of the root search and every node of the quadrature over x comes with its
# saddlepoint, and no saddlepoint equation is solved there. For a ratio the parameter is the
# saddlepoint ŝ of W = X_1 - rX_2 along the locus s K_1(s, t) + t K_2(s, t) = 0, t = -rŝ, on which
# r = K_1/K_2 (the counterpart of the locus of the Studentized mean).

# derivatives K^(k)(s), k = 0..N, of a univariate CGF, or nothing outside the domain
function _uni_derivs(K::CGF, s::T, N::Integer) where {T}
    P = _taylor(K, [s], N)
    P === nothing && return nothing
    Kd = T[_coef(T, P, (k,)) for k in 0:N]
    (all(isfinite, Kd) && Kd[3] > 0) || return nothing
    return Kd
end

# safeguarded Newton for an increasing function of s: ev(s) -> (; g, dg, ...) or nothing outside
# the domain. Steps are limited to 4 scale(e), pulled back into the domain by halving, and replaced
# by bisection once the root is bracketed and a step leaves the bracket
function _snewton(ev, s0::T, scale, tol::Real; maxit::Integer=200, name="quantile") where {T}
    s = s0; e = ev(s)
    e === nothing && error("$name: the starting point is outside the domain of K")
    lo, hi = T(-Inf), T(Inf)
    for it in 1:maxit
        abs(e.g) <= tol && return s, e, it
        e.g < 0 ? (lo = s) : (hi = s)
        smax = 4*scale(e)
        step = e.dg > 0 ? -e.g/e.dg : (e.g < 0 ? smax : -smax)
        sn = s + clamp(step, -smax, smax)
        isfinite(lo) && isfinite(hi) && !(lo < sn < hi) && (sn = (lo + hi)/2)
        en = ev(sn)
        while en === nothing                             # outside the domain: back toward s
            sn = (s + sn)/2
            abs(sn - s) <= 4eps(T)*max(abs(s), one(T)) &&
                error("$name: the iteration reached the boundary of the domain of K")
            en = ev(sn)
        end
        if isfinite(lo) && isfinite(hi) && hi - lo <= 8eps(T)*max(abs(lo), abs(hi), one(T))
            return sn, en, it                            # bracket at rounding level
        end
        s, e = sn, en
    end
    error("$name: the iteration did not converge")
end

"""
    quantile(K::CGF, p; order=3, lower=true, tol=1e-12) -> (; q, tau, F, iterations)

The quantile of the saddlepoint distribution function of order `order` of a univariate random
variable with cumulant generating function `K` (Hauschildt-Robinson expansion, [`tailprob`](@ref)):
the point `q` with ``P(X ≤ q) = p`` under the expansion (`lower = false`: ``P(X ≥ q) = p``, for
upper quantiles of small probability, which are then computed directly). The root search runs in
saddlepoint space: the point is parametrized by its saddlepoint, ``x = K'(s)``, so that each
iterate gives the point and its saddlepoint at once and no saddlepoint equation is solved; the
derivative of the distribution function in ``s`` is the density expansion of the same order times
``K''(s)``, and the Newton iteration is safeguarded by bracketing and bisection, with steps
limited to a few standard deviations and pulled back into the domain of ``K``. Converged when
the distribution function is within `tol` ``\\min(p, 1-p)`` of `p`; `tau` is the saddlepoint of `q`
and `F` the value of the expansion there. For continuous variables (the expansion of a lattice
variable is a step function of ``k``). The search runs in the floating-point type of `p`.

`quantile` is not the function of the standard library Statistics: with both packages loaded
write `SaddlepointExpansions.quantile`.
"""
function quantile(K::CGF, p::Real; order::Integer=3, lower::Bool=true, tol::Real=1e-12)
    K.d == 1 || throw(ArgumentError("quantile: K must be univariate (see ratio_quantile and studentized_quantile)"))
    _checkorder("quantile", order)
    0 < p < 1 || throw(ArgumentError("quantile: p must be in (0, 1)"))
    T = float(typeof(p)); p = T(p)
    N = 2order + 1
    function ev(s)
        Kd = _uni_derivs(K, s, N)
        Kd === nothing && return nothing
        r = spa_uni(Kd, s, Kd[2], order)
        F = lower ? r.cdf[end] : r.upp[end]
        return (; g=(lower ? F - p : p - F), dg=r.pdf[end]*Kd[3], x=Kd[2], F, K2=Kd[3])
    end
    s, e, it = _snewton(ev, zero(T), e -> 1/sqrt(e.K2), tol*min(p, 1 - p))
    return (; q=e.x, tau=s, F=e.F, iterations=it)
end

# the point of the saddlepoint locus of a ratio with abscissa s: t with s K_1 + t K_2 = 0, by
# Newton's method in t from tguess, with r = K_1/K_2 and the derivatives of K there (nothing if
# the iteration leaves the domain or does not converge)
function _ratio_locus(K::CGF, s::T, tguess::T) where {T}
    iszero(s) && (tguess = zero(T))
    t = tguess
    for it in 1:60
        P = _taylor(K, [s, t], 2)
        P === nothing && return nothing
        K1, K2 = _coef(T, P, (1, 0)), _coef(T, P, (0, 1))
        K11, K12, K22 = _coef(T, P, (2, 0)), _coef(T, P, (1, 1)), _coef(T, P, (0, 2))
        G = s*K1 + t*K2; Gt = s*K12 + K2 + t*K22
        (K2 > 0 && Gt > 0) || return nothing
        dt = G/Gt
        if abs(dt) <= 50eps(T)*(1 + abs(t)) || iszero(G)
            # r and its derivative along the locus, t' = -G_s/G_t
            Gs = K1 + s*K11 + t*K12
            tp = -Gs/Gt
            r = K1/K2
            rp = ((K11 + K12*tp)*K2 - K1*(K12 + K22*tp))/K2^2
            return (; t, r, rp, tp)
        end
        t -= dt
    end
    return nothing
end

# F and the density of the ratio at the point of the locus with abscissa s
function _ratio_at(K::CGF, s::T, loc, order::Integer) where {T}
    r = loc.r
    N = 2order + 1
    P = _taylor(K, [s, -r*s], N, T[1 0; -r 1])
    P === nothing && return nothing
    KWd = T[_coef(T, P, (k, 0)) for k in 0:N]
    (all(isfinite, KWd) && KWd[3] > 0) || return nothing
    gd = T[_coef(T, P, (l, 1)) for l in 0:2order-2]
    u = spa_uni(KWd, s, zero(T), order)
    f = spa_ratio_pdf(KWd[1:2order+1], gd, s, order)
    return (; u, f, KW2=KWd[3])
end

"""
    ratio_quantile(K::CGF, p; order=3, lower=true, tol=1e-12) -> (; r, shat, F, iterations)

The quantile of the saddlepoint distribution function of order `order` of the ratio
``R = X_1/X_2``, ``X_2 > 0``, with `K` the joint cumulant generating function of ``(X_1, X_2)``
([`ratio_tailprob`](@ref)): `r` with ``P(R ≤ r) = p`` under the expansion (`lower = false`:
``P(R ≥ r) = p``). The search runs in saddlepoint space, over the saddlepoint ``\\hat s`` of
``W = X_1 - rX_2``: the saddlepoints of all ``r`` lie on the curve ``sK_1(s,t) + tK_2(s,t) = 0``,
``t = -r\\hat s``, on which ``r = K_1/K_2``, so that each trial ``\\hat s`` gives its ``t``
(by a one-dimensional Newton iteration on the curve, continued from the previous iterate), its
``r`` and its distribution function directly, without the saddlepoint equation of ``W`` for a
trial ``r``. The Newton iteration in ``\\hat s`` uses the derivative ``f_R(r)\\,dr/d\\hat s``, with
the density expansion of the ratio of the same order ([`ratio_density`](@ref)) and ``dr/d\\hat s``
along the curve, safeguarded as in [`quantile`](@ref). `shat` is the saddlepoint and `F` the
value of the expansion at `r`.
"""
function ratio_quantile(K::CGF, p::Real; order::Integer=3, lower::Bool=true, tol::Real=1e-12)
    K.d == 2 || throw(ArgumentError("ratio_quantile: K must be the joint cumulant generating function of the numerator and the denominator (d = 2)"))
    _checkorder("ratio_quantile", order)
    0 < p < 1 || throw(ArgumentError("ratio_quantile: p must be in (0, 1)"))
    T = float(typeof(p)); p = T(p)
    last = Ref((zero(T), zero(T), zero(T)))               # (s, t, t') of the previous iterate
    function ev(s)
        s0, t0, tp0 = last[]
        loc = _ratio_locus(K, s, t0 + tp0*(s - s0))
        loc === nothing && return nothing
        a = _ratio_at(K, s, loc, order)
        a === nothing && return nothing
        last[] = (s, loc.t, loc.tp)
        F = lower ? a.u.cdf[end] : a.u.upp[end]
        return (; g=(lower ? F - p : p - F), dg=a.f[end]*loc.rp, r=loc.r, F, KW2=a.KW2)
    end
    s, e, it = _snewton(ev, zero(T), e -> 1/sqrt(e.KW2), tol*min(p, 1 - p); name="ratio_quantile")
    return (; r=e.r, shat=s, F=e.F, iterations=it)
end

# ----- the total mass of density expansions, by quadrature in saddlepoint space

# the interval of v (the line s = s0 + v e) where the integrand is not negligible: from v = 0, which
# must be inside the domain, steps doubling from h in each direction until logw(v) (nothing outside
# the domain) falls c below the largest value met, then bisection toward the boundary of the
# domain; an end at which the integrand still has not fallen after 2^12 h (an algebraic tail, as
# where a gamma variable approaches zero) is infinite
function _srange(logw, h::T, c::T) where {T}
    l0 = logw(zero(T))
    l0 === nothing && error("normalize: the starting point is outside the domain of K")
    ends = T[]
    for dir in (-1, 1)
        inside = zero(T); v = h; lmax = l0
        while true
            l = logw(dir*v)
            if l === nothing                              # outside: bisect toward the boundary
                out = v
                for _ in 1:60
                    mid = (inside + out)/2
                    lm = logw(dir*mid)
                    lm === nothing ? (out = mid) : (inside = mid)
                    (out - inside) <= 1e-12*max(out, one(T)) && break
                end
                push!(ends, dir*inside); break
            end
            lmax = max(lmax, l)
            if l < lmax - c
                push!(ends, dir*v); break
            end
            inside = v; v *= 2
            v > 4096h && (push!(ends, dir*T(Inf)); break)
        end
    end
    return ends[1], ends[2]
end

# the masses of the density expansions of orders 1..order of a CGF of dimension d, the integral
# over x of f_M(x) written as the integral over s of f_M(K'(s)) det K''(s), nested over the
# coordinates (the last outermost); every line starts from a point inside the domain (the
# regression of its coordinate on the outer ones under K''(0), or the nearest point inside along
# the line), and its range is where the leading term, for the inner lines, or the inner integral,
# for the outer ones, has not fallen by e^-50 from its largest value
function _density_mass(K::CGF, order::Integer, ::Type{T}; rtol::Real=(K.d == 1 ? 1e-12 : 1e-9)) where {T}
    d = K.d
    maxdeg = 2order + 1
    Kf = _kfun(K, maxdeg); Kw = _kwhite(K, maxdeg)
    e(j) = ntuple(i -> i == j ? 1 : 0, d)
    H0 = T[Kf(e(i) .+ e(j), zeros(T, d)) for i in 1:d, j in 1:d]
    all(isfinite, H0) || error("normalize: K is not finite at the origin")
    inside(s) = _indomain(K, s) && _taylor(K, s, 2) !== nothing
    function lead(s)                                     # log of f_1(K'(s)) det K''(s)
        P = _taylor(K, s, 2)
        P === nothing && return nothing
        g = T[_coef(T, P, e(j)) for j in 1:d]
        H = T[_coef(T, P, e(i) .+ e(j)) for i in 1:d, j in 1:d]
        ch = cholesky(Symmetric(H); check=false)
        (all(isfinite, H) && issuccess(ch)) || return nothing
        return _coef(T, P, ntuple(_ -> 0, d)) - dot(s, g) - (T(d)/2)*log(2*T(pi)) + sum(log.(diag(ch.L)))
    end
    function full(s)                                     # f_M(K'(s)) det K''(s), M = 1..order
        if d == 1
            Kd = _uni_derivs(K, s[1], maxdeg)
            Kd === nothing && return zeros(T, order)
            return spa_uni(Kd, s[1], Kd[2], order).pdf .* Kd[3]
        end
        P = _taylor(K, s, 2)
        P === nothing && return zeros(T, order)
        g = T[_coef(T, P, e(j)) for j in 1:d]
        H = T[_coef(T, P, e(i) .+ e(j)) for i in 1:d, j in 1:d]
        # where K'' is numerically singular (at the boundary of the domain, where the density is
        # negligible) the node counts as outside
        (all(isfinite, H) && issuccess(cholesky(Symmetric(H); check=false))) || return zeros(T, order)
        dn = _cond_dens(Kf, Kw, collect(s), g, order)
        return exp(dn.logf0) .* dn.s .* det(H)
    end
    c = T(50)
    # the integral over s_k (and the inner coordinates) with s[k+1:d] = outer
    function level(k, outer)
        ck = k == d ? zero(T) : -dot(H0[k, k+1:d], outer)/H0[k, k]
        pt(v) = vcat(zeros(T, k - 1), [v], outer)
        if !inside(pt(ck))                               # a point of the line inside the domain
            h = 1/sqrt(H0[k, k]); found = false
            for j in 0:40, dir in (-1, 1)
                v = ck + dir*h*T(2)^(j/2)
                inside(pt(v)) && (ck = v; found = true; break)
            end
            found || return zeros(T, order)
        end
        h = 1/sqrt(H0[k, k])
        if k == 1
            lw = v -> lead(pt(ck + v))
            f = v -> full(pt(ck + v))
        else
            memo = Dict{T,Vector{T}}()
            inner(v) = get!(() -> level(k - 1, vcat([ck + v], outer)), memo, v)
            lw = v -> (I = inner(v); I[1] > 0 ? log(I[1]) : nothing)
            f = inner
        end
        lo, hi = _srange(lw, h, c)
        return quadgk(f, lo, zero(T), hi; rtol, order=7)[1]
    end
    return level(d, T[])
end

# the mass of the ratio density expansions of orders 1..order: the integral over r of f_R(r)
# written as the integral over the saddlepoint ŝ of f_R(r(ŝ)) dr/dŝ along the locus
function _ratio_mass(K::CGF, order::Integer, ::Type{T}; rtol::Real=1e-10) where {T}
    cache = Dict{T,Tuple{T,T}}(zero(T) => (zero(T), zero(T)))
    function locus(s)
        # continue from the nearest abscissa solved so far
        sn = argmin(v -> abs(v - s), collect(keys(cache)))
        t0, tp0 = cache[sn]
        loc = _ratio_locus(K, s, t0 + tp0*(s - sn))
        if loc === nothing                                # in steps from the nearest one
            m = 8; tt, tp = t0, tp0
            for j in 1:m
                sj = sn + (s - sn)*j/m
                lj = _ratio_locus(K, sj, tt + tp*(s - sn)/m)
                lj === nothing && return nothing
                tt, tp = lj.t, lj.tp; loc = lj
            end
        end
        cache[s] = (loc.t, loc.tp)
        return loc
    end
    function vals(s)
        loc = locus(s)
        loc === nothing && return nothing
        a = _ratio_at(K, s, loc, order)
        a === nothing && return nothing
        return a.f .* loc.rp, a
    end
    function lw(s)
        v = vals(s)
        (v === nothing || !(v[1][1] > 0)) && return nothing
        return log(v[1][1])
    end
    a0 = vals(zero(T))
    a0 === nothing && error("normalize: the ratio expansion cannot be evaluated at the centre")
    lo, hi = _srange(lw, 1/sqrt(a0[2].KW2), T(50))
    f(s) = (v = vals(s); v === nothing ? zeros(T, order) : v[1])
    return quadgk(f, lo, zero(T), hi; rtol, order=7)[1]
end

# the mass of the Studentized density expansions of orders 1..order: the integral over q written
# as the integral over the saddlepoint ŝ along the locus s K_s + 2t K_t = 0, with dq/dŝ along it;
# the two sides of ŝ = 0 (where t = 0 and the expansion is not evaluated) separately
function _stud_mass(K::CGF, order::Integer, ::Type{T}; rtol::Real=1e-9) where {T}
    P0 = _stud_jet(K, zero(T), zero(T), 2)
    EU, EV = _coef(T, P0, (1, 0)), _coef(T, P0, (0, 1))
    slope = -EU/(2EV)                                     # t'(0) along the locus
    cache = Dict{T,T}()
    function vals(s)
        sn = isempty(cache) ? nothing : argmin(v -> abs(v - s), collect(keys(cache)))
        tg = (sn === nothing || abs(sn - s) > abs(s)) ? slope*s : cache[sn]
        r = try
            studentized_locus(K, s, tg)
        catch err
            err isa InterruptException && rethrow()
            return nothing
        end
        cache[s] = r.t
        P = _taylor(K, [s, r.t], 2)
        P === nothing && return nothing
        Ks, Kt = _coef(T, P, (1, 0)), _coef(T, P, (0, 1))
        Kss, Kst, Ktt = _coef(T, P, (2, 0)), _coef(T, P, (1, 1)), _coef(T, P, (0, 2))
        Gs = Ks + s*Kss + 2r.t*Kst; Gt = s*Kst + 2Kt + 2r.t*Ktt
        tp = -Gs/Gt
        vp = (Kst + Ktt*tp)/(2r.v)
        qp = ((Kss + Kst*tp)*r.v - Ks*vp)/r.v^2
        f = try
            studentized_density(K, r.q; order, saddle=(; s, t=r.t, v=r.v)).f
        catch err
            err isa InterruptException && rethrow()
            return nothing
        end
        return f .* qp
    end
    total = zeros(T, order)
    for dir in (-1, 1)
        s0 = dir*T(1//20)/sqrt(EV)
        v0 = vals(s0)
        v0 === nothing && error("normalize: the Studentized expansion cannot be evaluated near the centre")
        lw(v) = (val = vals(s0 + v); (val === nothing || !(val[1] > 0)) ? nothing : log(val[1]))
        lo, hi = _srange(lw, abs(s0)*4, T(50))
        f(v) = (val = vals(s0 + v); val === nothing ? zeros(T, order) : val)
        a, b = dir == -1 ? (lo, -s0) : (-s0, hi)          # the side of ŝ = 0, in v = s - s0
        a < b && (total .+= quadgk(f, a, b; rtol, order=7)[1])
    end
    return total
end
