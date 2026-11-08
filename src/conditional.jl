# Conditional saddlepoint expansions: the conditional distribution function
# P(X_A ≥ x_A | X_B = x_B) and the conditional density of X_A given X_B = x_B, for continuous and
# for lattice variables (Broda and Kan, "Conditional saddlepoint expansions of arbitrary order",
# Theorems 1 to 3). The numerator of the conditional probability is the orthant-probability
# integral with the poles in the components of A only, the denominator the density expansion of
# X_B at its own saddlepoint. The numerator is evaluated in the canonical coordinates of Section
# 5.2 of that paper,
#     Y_A = X_A - C X_B,  Y_B = L^{-1} X_B,  C = K''_AB (K''_BB)^{-1},  LL' = K''_BB,
# in which the tilted covariance matrix is block diagonal, diag(K''_{AA.B}, I); the conditional
# probability is unchanged, and the terms of the expansion no longer cancel. The density
# expansions are evaluated in fully whitened coordinates.

# A as a sorted vector of distinct indices in 1:d
function _cond_set(A, d::Integer, name)
    Av = A isa Integer ? [Int(A)] : collect(Int, A)
    (!isempty(Av) && allunique(Av) && all(1 .<= Av .<= d)) ||
        throw(ArgumentError("$name: A must be a nonempty set of distinct indices in 1:$d"))
    return sort(Av)
end

# the multi-index nu (of the components idx) embedded in dimension d
_embed_nu(nu, idx, d::Integer) = ntuple(j -> (i = findfirst(==(j), idx); i === nothing ? 0 : nu[i]), d)

# the multi-index tables, created once per (dimension, degree)
const _MINDEX = Dict{Tuple{Int,Int},MIndex}()
_mindex(d::Integer, N::Integer) = lock(_SPACELOCK) do
    get!(() -> MIndex(d, N), _MINDEX, (Int(d), Int(N)))
end

# indices of the multi-indices of degree k in the graded list of I
_degrange(I::MIndex, k::Integer) = (k == 0 ? 1 : mi_count(I, k - 1) + 1):mi_count(I, k)

# He_nu(0, I) = prod_j He_{nu_j}(0) on the list of I (He_n(0) = (-1)^(n/2) (n-1)!! for even n)
function _he0_list(::Type{T}, I::MIndex) where {T}
    h = hcoef(T, I.N ÷ 2 + 1)
    return T[prod(iseven(n) ? h[n÷2+1] : zero(T) for n in nu; init=one(T)) for nu in I.list]
end

# substitution of the variable in slot j by the linear form ell'u in the polynomial with the
# coefficients th on the list of I (all degrees are kept: the substitution is homogeneous)
function _polysub_slot(th::Vector{T}, I::MIndex{d}, j::Integer, ell::AbstractVector) where {T,d}
    supp = findall(!iszero, ell)
    ns = length(supp)
    fac = [factT(T, k) for k in 0:I.N]
    new = zeros(T, length(th))
    for q in eachindex(th)
        v = th[q]
        iszero(v) && continue
        nu = I.list[q]; m = nu[j]
        if m == 0
            new[q] += v; continue
        end
        ns == 0 && continue
        for c in compositions(m, ns)
            mu = Base.setindex(nu, 0, j)
            coef = fac[m+1]
            for (i, ci) in enumerate(c)
                mu = Base.setindex(mu, mu[supp[i]] + ci, supp[i])
                coef *= T(ell[supp[i]])^ci/fac[ci+1]
            end
            new[mi_rank(I, mu)] += v*coef
        end
    end
    return new
end

# c[r][k] = [u^nu_k] G^r/r!, r = 1..R, from the coefficients theta of G (degrees >= 3), computed
# only for 3r <= |nu| <= dmax(r): the terms of the expansions use no others
function _cond_coeffs(theta::Vector{T}, I::MIndex{d}, R::Integer, dmax) where {T,d}
    c = Vector{Vector{T}}(undef, max(R, 0))
    R >= 1 || return c
    c[1] = theta
    supp = sort([(I.list[k], theta[k], sum(I.list[k])) for k in eachindex(theta) if !iszero(theta[k])];
                by=last)
    for r in 2:R
        dm = dmax(r)
        A = zeros(T, length(theta)); prev = c[r-1]
        for k in 1:mi_count(I, dm - 3)
            v = prev[k]
            iszero(v) && continue
            nu = I.list[k]; dk = sum(nu)
            for (mu, th, dmu) in supp
                dk + dmu > dm && break
                A[mi_rank(I, nu .+ mu)] += th*v
            end
        end
        c[r] = A ./ r
    end
    return c
end

# Taylor coefficients theta[k] = [v^nu_k] K(tau + Tm v), 3 <= |nu_k| <= nmax, on the list of I:
# from Kwhite (the derivatives of K_Y(u) = K(Tm u) at uh, Tm uh = tau), or from the derivatives
# of K at tau by the substitutions subs = [(j, ell), ...], applied in turn: the variable in slot j
# is replaced by the linear form ell'u, which may involve only the variables of slots already
# substituted, or of slots that keep their variable
function _cond_theta(Kfun, Kwhite, tau::Vector{T}, uh::Vector{T}, Tm::Matrix{T}, I::MIndex{d},
                     nmax::Integer, subs) where {T,d}
    theta = zeros(T, length(I.list))
    for k in 1:mi_count(I, nmax)
        nu = I.list[k]
        sum(nu) >= 3 || continue
        f = _fact(T, nu)
        theta[k] = Kwhite === nothing ? T(Kfun(nu, tau))/f : T(Kwhite(nu, uh, Tm))/f
    end
    Kwhite === nothing || return theta
    for (j, ell) in subs
        theta = _polysub_slot(theta, I, j, ell)
    end
    return theta
end

# value and Hessian of K at tau from Kfun
function _cond_K02(Kfun, tau::Vector{T}) where {T}
    d = length(tau)
    e(j) = ntuple(i -> i == j ? 1 : 0, d)
    K0 = T(Kfun(ntuple(_ -> 0, d), tau))
    K2 = T[Kfun(e(i) .+ e(j), tau) for i in 1:d, j in 1:d]
    (isfinite(K0) && all(isfinite, K2)) || error("conditional: the saddlepoint is outside the domain of K")
    return K0, (K2 + K2')/2
end

# ----- the density expansion (Theorem 5 of the first paper) in whitened coordinates

# log of the leading term and the bracket 1 + sum_m n^-m ... of orders 1..order at the
# saddlepoint tau of x, with the cancellation indicators of the bracket
function _cond_dens(Kfun, Kwhite, tau::Vector{T}, x::Vector{T}, order::Integer) where {T}
    d = length(x)
    K0, K2 = _cond_K02(Kfun, tau)
    Lc = cholesky(Symmetric(K2)).L
    logf0 = K0 - dot(tau, x) - (T(d)/2)*log(2*T(pi)) - sum(log.(diag(Lc)))
    s = ones(T, order); kappa = ones(T, order)
    order == 1 && return (; logf0, s, kappa)
    Tm = Matrix{T}(inv(Matrix(Lc')))
    uh = Lc'*tau
    Lx = 6(order - 1)
    I = _mindex(d, Lx)
    theta = _cond_theta(Kfun, Kwhite, tau, uh, Tm, I, 2order, [(j, Tm[j, :]) for j in d:-1:1])
    R = 2(order - 1)
    c = _cond_coeffs(theta, I, R, r -> min(Lx, 2(order - 1) + 2r))
    He0 = _he0_list(T, I)
    acc = one(T); ab = one(T)
    for M in 2:order
        m = M - 1
        for r in 1:2m, k in _degrange(I, 2m + 2r)
            (iszero(c[r][k]) || iszero(He0[k])) && continue
            t = c[r][k]*He0[k]
            acc += t; ab += abs(t)
        end
        s[M] = acc; kappa[M] = ab/abs(acc)
    end
    return (; logf0, s, kappa)
end

# ----- the lattice amplitude g(s) = s/(2 sinh(s/2))

# coefficients of 1/Q(u), Q(u) = 2 sinh(u/2)/u = sum_k u^(2k)/(4^k (2k+1)!), degrees 0..n
function _ginv_series(::Type{T}, n::Integer) where {T}
    Q = zeros(T, n + 1)
    for k in 0:n÷2
        Q[2k+1] = one(T)/(T(4)^k*factT(T, 2k + 1))
    end
    co = zeros(T, n + 1); co[1] = one(T)
    for m in 1:n
        co[m+1] = -sum(Q[j+1]*co[m-j+1] for j in 1:m)
    end
    return co
end

# Taylor coefficients at s0, degrees 0..L, of the polynomial sum_k co[k+1] s^k
function _shiftpoly(co::Vector{T}, s0::T, L::Integer) where {T}
    g = zeros(T, L + 1)
    for k in 0:length(co)-1, j in 0:min(k, L)
        g[j+1] += co[k+1]*T(binomial(big(k), j))*s0^(k - j)
    end
    return g
end

# Taylor coefficients at s0, degrees 0..L, of g(s) = s/(2 sinh(s/2)) (mode :saddle), or of its
# expansion about the origin truncated after the term s^deg (mode :origin, Kolassa 2003)
function _sinhcoef(s0::T, L::Integer, mode::Symbol, deg::Integer=0) where {T}
    mode === :origin && return _shiftpoly(_ginv_series(T, deg), s0, L)
    if abs(s0) >= 1
        # g = s/q with q = 2 sinh(s/2), by series division
        q = [(iseven(k) ? 2*Base.sinh(s0/2) : 2*Base.cosh(s0/2))/(T(2)^k*factT(T, k)) for k in 0:L]
        pp = zeros(T, L + 1); pp[1] = s0; L >= 1 && (pp[2] = one(T))
        g = zeros(T, L + 1); g[1] = pp[1]/q[1]
        for m in 1:L
            g[m+1] = (pp[m+1] - sum(q[j+1]*g[m-j+1] for j in 1:m))/q[1]
        end
        return g
    end
    # the series at the origin (radius 2 pi), shifted to s0
    n = L + max(40, ceil(Int, -log(eps(T))/log(2*T(pi))) + 10)
    return _shiftpoly(_ginv_series(T, n), s0, L)
end

# ----- Gauss-Legendre nodes in the working precision

const _GLT = Dict{Tuple{Int,Int},Tuple{Vector{BigFloat},Vector{BigFloat}}}()

function _gauleg_T(::Type{T}, n::Integer) where {T}
    T === Float64 && return gauleg(n)
    if T === BigFloat
        key = (precision(BigFloat), Int(n))
        hit = lock(() -> get(_GLT, key, nothing), _QLOCK)
        hit === nothing || return hit
    end
    x0, _ = gauleg(n)
    xs = Vector{T}(undef, n); ws = Vector{T}(undef, n)
    for i in 1:n
        x = T(x0[i]); dp = one(T)
        for it in 1:100
            p0, p1 = one(T), x
            for k in 2:n
                p0, p1 = p1, ((2k - 1)*x*p1 - (k - 1)*p0)/k
            end
            dp = n*(x*p1 - p0)/(x^2 - 1)
            dx = p1/dp
            x -= dx
            abs(dx) <= 2eps(T) && it > 1 && break
        end
        p0, p1 = one(T), x
        for k in 2:n
            p0, p1 = p1, ((2k - 1)*x*p1 - (k - 1)*p0)/k
        end
        dp = n*(x*p1 - p0)/(x^2 - 1)
        xs[i] = x; ws[i] = 2/((1 - x^2)*dp^2)
    end
    T === BigFloat && lock(() -> (_GLT[(precision(BigFloat), Int(n))] = (xs, ws)), _QLOCK)
    return xs, ws
end

# ----- the integrals over the orthant of the tail components

# nodes v >= 0 and weights of three Gauss-Legendre panels for the integrands
# He_k(v) exp(-(u + v)^2/2), k <= L, on [0, inf), truncated where L log(max(v, 1)) - (u + v)^2/2
# falls c below its maximum (for L = 0 and u >= 0 this is the truncation u vmax + vmax^2/2 = c of
# the paper; the factor v^L, which the paper's rule leaves out, moves the end point out for the
# integrals of high degree); for u >= 0 the panels are split at 1/(1 + u) and 4/(1 + u), where
# exp(-u v) decays, and otherwise around the mode v = -u
function _cond_panels(u::T, G::Integer, c::T, L::Integer) where {T}
    xg, wg = _gauleg_T(T, G)
    h(v) = L*log(max(v, one(T))) - (u + v)^2/2
    vq = max(zero(T), -u)                                # maximizer of the Gaussian factor
    vs = L > 0 ? (-u + sqrt(u^2 + 4L))/2 : zero(T)       # stationary point where v >= 1
    vst = vs > 1 ? max(vq, vs) : vq                      # h decreases beyond vst
    hm = max(h(vq), vs > 1 ? h(vs) : h(vq))
    lo, hi = vst, vst + 1
    while h(hi) > hm - c
        lo, hi = hi, 2hi
    end
    for _ in 1:200                                       # bisection for the end point
        mid = (lo + hi)/2
        h(mid) > hm - c ? (lo = mid) : (hi = mid)
        hi - lo <= 8eps(T)*hi && break
    end
    vmax = hi
    if u >= 0
        br = (zero(T), min(1/(1 + u), vmax/3), min(4/(1 + u), 2vmax/3), vmax)
    else
        v0 = max(zero(T), -u - sqrt(2c))
        b1 = clamp(-u - 2, v0, vmax); b2 = clamp(-u + 2, b1, vmax)
        br = (v0, b1, b2, vmax)
    end
    v = T[]; w = T[]
    for k in 1:3
        hw = (br[k+1] - br[k])/2
        hw > 0 || continue
        m = (br[k] + br[k+1])/2
        append!(v, m .+ hw .* xg); append!(w, hw .* wg)
    end
    return v, w
end

# The integrals over the orthant z_A >= t_A of the tail components (t_A = S tA, S = K''_{AA.B},
# z_A = t_A + w) in the canonical coordinates, the conditioning components fixed at t_B, in the
# form of Appendix B of the first paper: IA[k] = exp(logE0) ∫_{w >= 0} He_{mu_k}(v) ×
# exp(-tA'w - w'S^{-1}w/2) dw/((2π)^{a/2}|S|^{1/2}), mu_k = IA.list[k], with
# He_mu(v) = prod_i He_{mu_i}(v_i) in the whitened variable v = LA^{-1} w, LA LA' = S (the Taylor
# coefficients are taken in the same whitened coordinates; for a = 1 this is a scaling). The
# exponential factor E_n is combined with the normal density in the log domain. G and G/2 nodes
# per panel (the second for the error estimate); for a = 1 in other types than Float64, the
# forward recursion of I_k(u) = ∫_0^∞ He_k(v) φ(u + v) dv instead (exact; the error estimate is
# then zero).
function _cond_IA(IA::MIndex, S::Matrix{T}, LA::AbstractMatrix{T}, tA::Vector{T}, logE0::T,
                  G::Integer) where {T}
    a = length(tA); L = IA.N
    c = T === Float64 ? T(40) : max(T(40), -log(eps(T)) + 10)
    if a == 1
        sd = LA[1, 1]; u = sd*tA[1]
        if T !== Float64
            R = _cond_I1rec(u, L)
            v = [exp(logE0)*R[k+1]/sqrt(2*T(pi)) for k in 0:L]
            return v, copy(v), true
        end
        function run1(G)
            vv, ww = _cond_panels(u, G, c, L)
            W = ww .* exp.(logE0 .- u .* vv .- vv.^2 ./ 2) ./ sqrt(2*T(pi))
            out = zeros(T, L + 1)
            h0 = ones(T, length(vv)); h1 = copy(vv)
            out[1] = sum(W)
            L >= 1 && (out[2] = dot(W, h1))
            for k in 1:L-1
                h0, h1 = h1, vv .* h1 .- k .* h0
                out[k+2] = dot(W, h1)
            end
            return out
        end
        return run1(G), run1(max(8, G ÷ 2)), false
    end
    Si = inv(S); Si = (Si + Si')/2
    ldS = 2sum(log.(diag(LA)))
    Ia = Matrix{T}(LinearAlgebra.I, a, a)
    function runa(G)
        W, lq = _cond_grid(S, tA, G, c, L)
        n = size(W, 2)
        lw = lq .+ logE0 .- vec(tA'*W) .- vec(sum(W .* (Si*W); dims=1)) ./ 2 .- (T(a)/2)*log(2*T(pi)) .- ldS/2
        wt = exp.(lw)
        V = LowerTriangular(LA) \ W
        out = zeros(T, length(IA.list))
        chunk = max(64, floor(Int, 4e6/length(IA.list)))
        for c0 in 1:chunk:n
            c1 = min(n, c0 + chunk - 1)
            He = hermite_mv(IA, Ia, V[:, c0:c1])
            out .+= He'*wt[c0:c1]
        end
        return out
    end
    return runa(G), runa(max(8, G ÷ 2)), false
end

# I_k(u)/φ(u), k = 0..L, by the forward recursion R_{k+1} = He_k(0) - u R_k, R_0 = Φ(-u)/φ(u),
# in raised precision for BigFloat (the recursion loses about log10(u^k) digits at index k)
function _cond_I1rec(u::T, L::Integer) where {T}
    function rec(u)
        S = typeof(u)
        h = hcoef(S, L ÷ 2 + 1)
        R = zeros(S, L + 1)
        R[1] = normcdf(-u)/normpdf(u)
        for k in 0:L-1
            R[k+2] = (iseven(k) ? h[k÷2+1] : zero(S)) - u*R[k+1]
        end
        return R
    end
    T === BigFloat || return rec(u)
    p = precision(BigFloat)
    bits = p + ceil(Int, L*log2(1 + abs(Float64(u)))) + 32
    R = setprecision(BigFloat, bits) do
        rec(BigFloat(u; precision=bits))
    end
    return [BigFloat(r; precision=p) for r in R]
end

# nodes w = z_A - t_A >= 0 of the a-dimensional orthant in sequentially conditioned coordinates of
# N(0, S), t_A = S tA, with the logarithms of the products of the panel weights (dz). The first
# coordinate has the panels of the case a = 1 with G nodes each; each later one, given the
# earlier, three panels placed by its own standardized truncation point (t_i - mu_i)/s_i, with G
# nodes each for a = 2 and max(8, G/2) for more tail components.
function _cond_grid(S::Matrix{T}, tA::Vector{T}, G::Integer, c::T, L::Integer) where {T}
    a = length(tA)
    t = S*tA
    s1 = sqrt(S[1, 1])
    v, w = _cond_panels(t[1]/s1, G, c, L)
    Z = reshape(t[1] .+ s1 .* v, 1, :)
    lq = log.(s1 .* w)
    for i in 2:a
        Sp = S[1:i-1, 1:i-1]
        beta = vec(S[i:i, 1:i-1] / Sp)
        si = sqrt(S[i, i] - dot(beta, S[1:i-1, i]))
        Gi = a == 2 ? G : max(8, G ÷ 2)
        Zn = Vector{Vector{T}}(); lqn = T[]
        for q in 1:size(Z, 2)
            mu = dot(beta, Z[:, q])
            vi, wi = _cond_panels((t[i] - mu)/si, Gi, c, L)
            for (vv, ww) in zip(vi, wi)
                push!(Zn, vcat(Z[:, q], t[i] + si*vv))
                push!(lqn, lq[q] + log(si*ww))
            end
        end
        Z = reduce(hcat, Zn); lq = lqn
    end
    return Z .- t, lq
end

# ----- the numerator of the conditional tail probability (Theorems 1 and 3)

# E_n [sum_{m <= j} sum_r sum_|nu|=m+2r ct_{r,nu} I_A(nu)], j = 0..2order-1, at the saddlepoint
# tau of the (continuity-corrected) point x, all tau_A >= 0, scaled by exp(-logscale); with the
# sums of the absolute values of the terms and the change with half the quadrature nodes. The
# coordinates are the canonical ones of Section 5.2 of the paper, s_A = u'_A,
# s_B = -C'u'_A + LB^{-T}u_B, followed by the whitening u'_A = LA^{-T}u_A of the tail components,
# LA LA' = K''_{AA.B}, with which the integrals are evaluated as in Appendix B of the first paper
# (for one tail component a scaling, which changes nothing)
function _cond_numer(Kfun, Kwhite, tau::Vector{T}, x::Vector{T}, A::Vector{Int}, order::Integer,
                     logscale::T, lattice::Bool, mode::Symbol, nG::Integer, sinhdeg) where {T}
    d = length(x); B = setdiff(1:d, A); a = length(A); b = length(B)
    p = 2order - 1; L = 3p
    K0, K2 = _cond_K02(Kfun, tau)
    T1 = Matrix{T}(LinearAlgebra.I, d, d)
    uh = copy(tau)
    if b > 0
        KBB = Symmetric(K2[B, B])
        Lb = cholesky(KBB).L
        C = K2[A, B]/Matrix(KBB)
        T1[B, A] = -C'
        T1[B, B] = inv(Matrix(Lb'))
        SA = K2[A, A] - C*K2[B, A]
        uh[B] = Lb'*(tau[B] + C'*tau[A])
        logdetL = sum(log.(diag(Lb)))
    else
        SA = copy(K2); logdetL = zero(T)
    end
    SA = (SA + SA')/2
    chA = cholesky(Symmetric(SA); check=false)
    issuccess(chA) || error("conditional: K''_{AA.B} is not positive definite at the saddlepoint")
    LA = Matrix(chA.L)
    UA = inv(Matrix(LA'))
    T2 = Matrix{T}(LinearAlgebra.I, d, d); T2[A, A] = UA
    Tm = T1*T2
    uh[A] = LA'*tau[A]
    IL = _mindex(d, L); nN = length(IL.list)
    # coefficients of G^r/r! in the coordinates s = Tm u
    subs = vcat([(j, T1[j, :]) for j in reverse(B)], [(j, T2[j, :]) for j in reverse(A)])
    theta = _cond_theta(Kfun, Kwhite, tau, uh, Tm, IL, p + 2, subs)
    c = _cond_coeffs(theta, IL, p, r -> min(L, p + 2r))
    delta = zeros(T, nN); delta[1] = one(T)
    if !lattice
        ct = vcat([delta], c)
        ctM = [ct for _ in 1:order]
    else
        # the amplitude prod_{j in A} g(s_j): its Taylor coefficients at tau in s_A, in the whitened
        # variables u_A (s_A = tau_A + UA (u_A - uh_A)) by the same substitution
        ctM = Vector{Vector{Vector{T}}}(undef, order)
        for M in 1:order
            if mode === :saddle && M > 1
                ctM[M] = ctM[1]; continue
            end
            dg = mode === :origin ? (sinhdeg === nothing ? 2M - 2 : sinhdeg) : 0
            G = [_sinhcoef(tau[A[i]], L, mode, dg) for i in 1:a]
            ctM[M] = _cond_lattice_ct(c, G, IL, A, UA, p, L)
        end
    end
    # the integrals I_A(nu) = I_A(nu_A) He_{nu_B}(0): under the Gaussian of the integrand the tail
    # and the conditioning components are independent in these coordinates
    logE0 = K0 - dot(tau, x) - logdetL - (T(b)/2)*log(2*T(pi)) - logscale
    IA = _mindex(a, L)
    Iv, Ih, exact = _cond_IA(IA, SA, LA, tau[A], logE0, nG)
    hB = hcoef(T, L ÷ 2 + 1)
    function full(Iv)
        out = zeros(T, nN)
        for (k, nu) in enumerate(IL.list)
            hb = one(T)
            for j in B
                isodd(nu[j]) && (hb = zero(T); break)
                hb *= hB[nu[j]÷2+1]
            end
            iszero(hb) && continue
            out[k] = hb*Iv[mi_rank(IA, ntuple(i -> nu[A[i]], a))]
        end
        return out
    end
    Inu = full(Iv)
    # partial sums over m <= j, j = 0..p, each with the amplitude of the order M = j÷2 + 1
    function sums(Inu)
        F = zeros(T, p + 1); ab = zeros(T, p + 1)
        for j in 0:p
            ct = ctM[j÷2+1]; s = zero(T); sa = zero(T)
            for m in 0:j, r in 0:m, k in _degrange(IL, m + 2r)
                v = ct[r+1][k]
                (iszero(v) || iszero(Inu[k])) && continue
                tt = v*Inu[k]
                s += tt; sa += abs(tt)
            end
            F[j+1] = s; ab[j+1] = sa
        end
        return F, ab
    end
    D, Dabs = sums(Inu)
    err = exact ? zeros(T, p + 1) : abs.(sums(full(Ih))[1] .- D)
    return (; D, Dabs, err)
end

# ct_r = sum_mu c_{r, . - mu} gamma_mu, r = 0..p, with c_0 = delta_0 and gamma the Taylor
# coefficients of h(s) = prod_{j in A} g(s_j) at tau (G[i] those of g at tau_{A_i}), expressed in
# the whitened variables of the tail components, s_A - tau_A = UA (u_A - uh_A) (multi-indices
# supported on A)
function _cond_lattice_ct(c::Vector{Vector{T}}, G::Vector{Vector{T}}, IL::MIndex{d}, A::Vector{Int},
                          UA::Matrix{T}, p::Integer, L::Integer) where {T,d}
    a = length(A)
    IA = MIndex(a, L)
    h = T[prod(G[i][mu[i]+1] for i in 1:a) for mu in IA.list]
    for i in a:-1:1                                      # s_{A_i} -> sum_{k >= i} UA[i, k] u_{A_k}
        h = _polysub_slot(h, IA, i, UA[i, :])
    end
    gam = Tuple{NTuple{d,Int},T,Int}[]
    for (k, mu) in enumerate(IA.list)
        iszero(h[k]) && continue
        push!(gam, (_embed_nu(mu, A, d), h[k], sum(mu)))
    end
    nN = length(IL.list)
    ct = Vector{Vector{T}}(undef, p + 1)
    for r in 0:p
        dm = r == 0 ? p : min(L, p + 2r)
        v = zeros(T, nN)
        if r == 0
            for (mu, g, dmu) in gam
                dmu <= dm && (v[mi_rank(IL, mu)] += g)
            end
        else
            prev = c[r]
            for k in 1:mi_count(IL, dm)
                w = prev[k]
                iszero(w) && continue
                nu = IL.list[k]; dk = sum(nu)
                for (mu, g, dmu) in gam
                    dk + dmu > dm && break
                    v[mi_rank(IL, nu .+ mu)] += g*w
                end
            end
        end
        ct[r+1] = v
    end
    return ct
end

# ----- the tail probability, with the inclusion-exclusion over the negative components of A

# P = D/D_M with the numerator terms m <= j, j = 0..2order-1, and M = j÷2 + 1 terms in the
# denominator (D scaled by the leading density term of X_B, den.logf0), the sums of the absolute
# values of the terms (divided by |D_M|) and the error estimates; when the
# saddlepoint has negative components in A, by the inclusion-exclusion (Remark 1 of the paper):
# every term is the conditional probability, given the same X_B, of the components of A that
# are kept, those in T negated, with the components of N \ T marginalized
function _cond_tail(Kfun, Kwhite, x::Vector{T}, A::Vector{Int}, order::Integer, den, tau0,
                    lattice::Bool, mode::Symbol, nG::Integer, sinhdeg) where {T}
    d = length(x)
    xc = copy(x)
    lattice && (xc[A] .-= one(T)/2)
    tau = try
        spa_saddle(Kfun, xc, tau0 === nothing ? zeros(T, d) : T.(tau0))[1]
    catch err
        (lattice && !(err isa InterruptException)) || rethrow()
        # no saddlepoint at k - 1/2: the corrected point is not inside the convex hull of the
        # support, so that the probability is zero or one (P(S >= 0) for a count, for example)
        throw(DomainError(xc, "conditional: no saddlepoint at the continuity-corrected point; it is outside the interior of the convex hull of the support (the probability is 0 or 1 there)"))
    end
    Nn = [j for j in A if tau[j] < 0]
    if isempty(Nn)
        r = _cond_numer(Kfun, Kwhite, tau, xc, A, order, den.logf0, lattice, mode, nG, sinhdeg)
        sj = [den.s[j÷2+1] for j in 0:2order-1]
        return (; P=r.D ./ sj, abssum=r.Dabs ./ abs.(sj), err=r.err ./ abs.(sj), tau)
    end
    B = setdiff(1:d, A); Apos = setdiff(A, Nn)
    P = zeros(T, 2order); abssum = zeros(T, 2order); err = zeros(T, 2order)
    for mask in 0:2^length(Nn)-1
        Tset = [Nn[i] for i in eachindex(Nn) if (mask >> (i - 1)) & 1 == 1]
        sgn = isodd(length(Tset)) ? -1 : 1
        tail = sort(vcat(Apos, Tset))
        if isempty(tail)
            P .+= sgn; abssum .+= 1
            continue
        end
        keep = sort(vcat(tail, B)); k = length(keep)
        flip = ones(Int, d); flip[Tset] .= -1
        fk = flip[keep]
        Km = (nu, t) -> prod(fk[i]^nu[i] for i in 1:k; init=1)*Kfun(_embed_nu(nu, keep, d), flip .* _embed(t, keep, d))
        Kw = nothing
        if Kwhite !== nothing
            Bm = zeros(d, k)
            for (cc, j) in enumerate(keep)
                Bm[j, cc] = flip[j]
            end
            Kw = (nu, u, M) -> Kwhite(nu, u, Bm*M)              # K_B(M u) = K(B M u)
        end
        x2 = flip .* x
        lattice && (x2[Tset] .+= 1)                              # X_j < x_j  <=>  -X_j >= -x_j + 1
        A2 = [findfirst(==(j), keep) for j in tail]
        r = _cond_tail(Km, Kw, x2[keep], A2, order, den, (flip .* tau)[keep], lattice, mode, nG, sinhdeg)
        P .+= sgn .* r.P; abssum .+= r.abssum; err .+= r.err
    end
    return (; P, abssum, err, tau)
end

function _spa_cond(Kfun, Kwhite, x::AbstractVector, A, order::Integer; tail::Bool=true,
                   density::Bool=true, lattice::Bool=false, sinh::Symbol=:saddle, nG::Integer=64,
                   tau0=nothing, sinhdeg=nothing)
    order >= 1 || throw(ArgumentError("spa_cond: order must be a positive integer"))
    isempty(x) && throw(ArgumentError("spa_cond: x must be a nonempty vector"))
    all(isfinite, x) || throw(ArgumentError("spa_cond: x must be finite"))
    sinh in (:saddle, :origin) || throw(ArgumentError("spa_cond: sinh must be :saddle or :origin"))
    nG >= 8 || throw(ArgumentError("spa_cond: nG must be at least 8"))
    d = length(x)
    A = _cond_set(A, d, "spa_cond")
    T = float(tau0 === nothing ? eltype(x) : promote_type(eltype(x), eltype(tau0)))
    x = T.(collect(x))
    B = setdiff(1:d, A); b = length(B)
    # the denominator: the density expansion of X_B at its own saddlepoint
    if b > 0
        KB = (nu, s) -> Kfun(_embed_nu(nu, B, d), _embed(s, B, d))
        KwB = nothing
        if Kwhite !== nothing
            EB = zeros(d, b)
            for (cc, j) in enumerate(B)
                EB[j, cc] = 1.0
            end
            KwB = (nu, u, M) -> Kwhite(nu, u, EB*M)
        end
        taut, _ = spa_saddle(KB, x[B], tau0 === nothing ? zeros(T, b) : T.(collect(tau0))[B])
        den = _cond_dens(KB, KwB, taut, x[B], order)
        tautf = zeros(T, d); tautf[B] = taut
    else
        den = (; logf0=zero(T), s=ones(T, order), kappa=ones(T, order))
        tautf = zeros(T, d)
    end
    status = "ok"
    canc(v) = !all(isfinite, v) || maximum(v)*eps(T) > 1e-8
    canc(den.kappa) && (status = addstatus(status, "large cancellation in the denominator"))
    f = fill(T(NaN), order); kappaf = fill(T(NaN), order); tauf = fill(T(NaN), d)
    if density
        tauf, _ = spa_saddle(Kfun, x, tau0 === nothing ? zeros(T, d) : T.(collect(tau0)))
        num = _cond_dens(Kfun, Kwhite, tauf, x, order)
        f = exp(num.logf0 - den.logf0) .* num.s ./ den.s
        kappaf = num.kappa
        canc(kappaf) && (status = addstatus(status, "large cancellation in the density"))
    end
    P = fill(T(NaN), order); kappaP = fill(T(NaN), order); err = fill(T(NaN), order)
    Pp = fill(T(NaN), 2order); tau = fill(T(NaN), d)
    if tail
        t0 = density ? tauf : (tau0 === nothing ? nothing : T.(collect(tau0)))
        r = _cond_tail(Kfun, Kwhite, x, A, order, den, t0, lattice, sinh, nG, sinhdeg)
        Pp = r.P; tau = r.tau
        P = Pp[2:2:end]; err = r.err[2:2:end]
        kappaP = r.abssum[2:2:end] ./ abs.(P)
        canc(kappaP) && (status = addstatus(status, "large cancellation in the numerator"))
        all(isfinite, err) && err[order] > 1e-8*abs(P[order]) &&
            (status = addstatus(status, "integration not resolved"))
    end
    return (; P, f, tau, tauf, taut=tautf, kappaP, kappaf, kappaB=den.kappa, err, Pp, status)
end

"""
    spa_cond(Kfun, x, A, order; lattice=false, sinh=:saddle, nG=64, tau0=nothing, Kwhite=nothing)
        -> (; P, f, tau, tauf, taut, kappaP, kappaf, kappaB, err, Pp, status)

The conditional expansions of Broda and Kan, "Conditional saddlepoint expansions of arbitrary
order", of orders `1:order`, from a function `Kfun(nu, t)` returning the derivative of order `nu`
(a `d`-tuple of integers) of the cumulant generating function at `t` (`NaN` outside the domain),
at the point `x` (`d = length(x)`); `A` holds the indices of the components whose tail is wanted,
and the others, ``B``, are conditioned on. The counterpart of `spa_cond.m` of the paper.

* `P[M]`: ``P(X_A ≥ x_A ∣ X_B = x_B)`` of order `M` (Theorem 1): the numerator with the terms
  ``m ≤ 2M-1`` over the density expansion of ``X_B`` with `M` terms, relative error
  ``O(n^{-M})``. When the saddlepoint has negative components in ``A`` the inclusion-exclusion of
  Remark 1 of the paper is applied: each term is the conditional probability of the components of
  ``A`` that are kept, some negated, given the same ``X_B``, with the others marginalized, and its
  own saddlepoint.
* `f[M]`: the conditional density ``f(x_A ∣ x_B)`` of order `M` (Theorem 2), the ratio of the
  density expansions of ``X`` and of ``X_B`` with the terms ``m ≤ M-1``.
* `tau`: the saddlepoint of ``K`` at `x` (at the continuity-corrected point in the lattice case),
  `tauf` the saddlepoint at `x` used by the density, `taut` the saddlepoint of the marginal of
  ``X_B``, embedded as ``(0_A, \\tilde s_B)``.
* `kappaP`, `kappaf`, `kappaB`: cancellation indicators (sum of the absolute values of the terms
  over the absolute value of their sum) of the numerator of `P` (the terms of the
  inclusion-exclusion included), of the density expansion of ``X`` and of that of ``X_B``.
* `err[M]`: the change of `P[M]` with half the quadrature nodes (zero where the integrals are
  computed exactly); a comparison of two quadratures, not a bound, and not the asymptotic error.
* `Pp[j+1]`, ``j = 0, …, 2\\,\\mathrm{order}-1``: the expansions with the terms of the numerator up
  to ``n^{-j/2}`` and ``⌊j/2⌋ + 1`` terms of the denominator (`P[M] = Pp[2M]`); `Pp[1]` is the
  leading term, for one tail component the conditional analogue
  ``(\\hat u/\\hat z)e^{(\\hat u^2-\\hat w^2)/2}\\bar Φ(\\hat u)`` of the Hauschildt-Robinson
  approximation in the notation of Skovgaard (1987).
* `status`: `"ok"` or one or more of `"large cancellation in the numerator"` (or `"in the
  density"`, `"in the denominator"`: an indicator above ``10^{-8}/ε``) and `"integration not
  resolved"` (`err[order]` above ``10^{-8}|P[order]|``), joined by `"; "`.

With `lattice = true` the components in ``A`` are integer-valued and so is `x[A]` (Theorem 3):
`P` is ``P(S_A ≥ k_A ∣ S_B = k_B)`` with the continuity correction (the saddlepoint of the
numerator is at ``k_A - 1/2``) and the amplitude ``\\prod_{j∈A} s_j/\\{2\\sinh(s_j/2)\\}``, and `f` is
``P(S_A = k_A ∣ S_B = k_B)`` (the same formula as the density). The components in ``B`` may be
lattice or continuous. `sinh = :saddle` expands the amplitude about the saddlepoint (Section 4 of
the paper); `sinh = :origin` expands ``s/\\{2\\sinh(s/2)\\}`` about zero as
``1 - s^2/24 + 7s^4/5760 - …``, keeping the terms up to ``s^{2M-2}`` in the approximation of order
``M``, as in Kolassa (2003, 2006).

Computation (Section 5 of the paper): the Taylor coefficients of ``K`` at the saddlepoint are
taken in the canonical coordinates ``Y_A = X_A - CX_B``, ``Y_B = L^{-1}X_B``,
``C = K''_{AB}(K''_{BB})^{-1}``, ``LL' = K''_{BB}``, in which the tilted covariance matrix is block
diagonal, ``\\mathrm{diag}(K''_{AA⋅B}, I)``; the conditional probability is unchanged, and the
terms cancel far less (by up to sixteen orders of magnitude in the examples of the paper). The
coefficients ``c_{r,ν}`` follow from the recursion (3), and the integrals ``I_A(ν)`` factor into
the products of ``\\mathrm{He}_{ν_B}(0)`` and an integral over the orthant of the tail components,
which is one-dimensional for one tail component, whatever the number of conditioning variables:
three Gauss-Legendre panels of `nG` nodes in the standardized variable ``v``, split at
``1/(1 + \\hat u)`` and ``4/(1 + \\hat u)``, on ``[0, v_{\\max}]``, where ``v_{\\max}`` is the point
beyond which ``v^L e^{-\\hat uv - v^2/2}``, ``L = 3(2\\,\\mathrm{order} - 1)`` the highest degree of the
Hermite polynomials, has fallen by ``e^{-40}`` from its maximum (the paper's rule
``\\hat uv_{\\max} + v_{\\max}^2/2 = 40`` leaves out the factor ``v^L``, which costs up to ten digits
in the integrals of high degree when ``\\hat u`` is small), and the same with half the nodes for
`err`. For more tail components the orthant integral is computed in sequentially conditioned
coordinates, each later coordinate with three panels placed by its own conditional truncation
point (`nG` nodes each for two tail components, `nG/2` for more), and the tail components are whitened as well, ``K''_{AA⋅B} = L_AL_A'``: the
coefficients are taken in the whitened coordinates and the Hermite polynomials are products of
univariate ones in ``L_A^{-1}(z_A - t_A)``, integrated over the same orthant (Appendix B of the
first paper), which removes the cancellation that strongly correlated tail components cause; for
one tail component this is a scaling and changes nothing. The density expansions are evaluated in
whitened coordinates. `Kwhite` is an optional function `(nu, u, M) ->` derivative of order `nu` at
`u` of ``K_Y(u) = K(Mu)`` for a ``d×k`` matrix `M` (as for [`spa_orthant`](@ref)); without it the
coefficients in the canonical and whitened coordinates are obtained by substituting the linear
maps into the Taylor polynomial of ``K`` at the saddlepoint, which loses digits when ``K''`` is ill
conditioned. [`conditional_tailprob`](@ref) and [`conditional_density`](@ref) supply `Kwhite` from
the Taylor-mode derivatives of a [`CGF`](@ref).

The arithmetic is done in the floating-point type of `x`; for one tail component and types other
than `Float64` the orthant integrals are computed exactly by the forward recursion
``I_{k+1}(u) = \\mathrm{He}_k(0)φ(u) - uI_k(u)`` (in raised precision for `BigFloat`), and for more
tail components by quadrature in the working precision. `tau0` is a starting value for the
saddlepoint iterations.
"""
function spa_cond(Kfun, x::AbstractVector, A, order::Integer; Kwhite=nothing, kwargs...)
    return _spa_cond(Kfun, Kwhite, x, A, order; kwargs...)
end

# the sign vector of the tail components for lower and mixed tails (ones on B)
function _cond_signs(name, d, A, lower, signs)
    sg = ones(Int, d)
    if signs === nothing
        lower && (sg[A] .= -1)
        return sg
    end
    length(signs) == length(A) ||
        throw(ArgumentError("$name: signs must have one entry per tail component ($(length(A)))"))
    all(s -> s == 1 || s == -1, signs) || throw(ArgumentError("$name: signs must be ±1"))
    sg[A] .= collect(Int, signs)
    return sg
end

"""
    conditional_tailprob(K::CGF, x, A; order=3, lower=false, signs=nothing, lattice=false,
                         sinh=:saddle, nG=64) -> (; F, tau, taut, status, info)

Saddlepoint expansions of orders `1:order` of the conditional distribution function
``P(X_A ≥ x_A ∣ X_B = x_B)`` (componentwise in ``A``) for the random vector with cumulant
generating function `K`, at the point `x` (a vector of length `d`). `A` holds the indices of the
components whose tail is wanted (an integer or a vector); the others, ``B``, are conditioned on
(Broda and Kan, "Conditional saddlepoint expansions of arbitrary order", Theorem 1). `F[M]` is
the expansion of order `M`: the numerator, the orthant-probability integral with the poles in
the components of ``A`` only, with the terms of orders ``n^{-m/2}``, ``m ≤ 2M-1``, over the
density expansion of ``X_B`` with `M` terms; its relative error is ``O(n^{-M})`` (the "``n``" is
absorbed in `K`). With one tail component, `F[1]` adds the terms of order ``n^{-1/2}`` to the
conditional analogue ``(\\hat u/\\hat z)e^{(\\hat u^2 - \\hat w^2)/2}\\bar Φ(\\hat u)`` of the
Hauschildt-Robinson approximation, in the notation of Skovgaard (1987). With ``B`` empty the
expansion is that of [`tailprob`](@ref).

* `lower = true` gives ``P(X_A ≤ x_A ∣ X_B = x_B)``, and `signs` (one ``±1`` per component of
  ``A``) a mixed corner; the tail components are reflected ([`reflect`](@ref)).
* When the saddlepoint has negative components in ``A`` the expansion is applied through the
  inclusion-exclusion identity of Remark 1 of the paper, every term with its own saddlepoint.
* `lattice = true`: the components in ``A`` are integer-valued and so is `x[A]`; `F` is then
  ``P(S_A ≥ k_A ∣ S_B = k_B)`` by Theorem 3, with the continuity correction and the amplitude
  ``\\prod_{j∈A}s_j/\\{2\\sinh(s_j/2)\\}`` expanded about the saddlepoint (`sinh = :saddle`) or about
  the origin (`sinh = :origin`, Kolassa 2003).
* `tau` is the saddlepoint of `K` at `x` (at the corrected point for lattice variables), `taut`
  that of the marginal of ``X_B``, embedded as ``(0_A, \\tilde s_B)``. `status` is `"ok"` or a
  description of a problem (see [`spa_cond`](@ref)); `info` holds `Fp` (the expansions with the
  terms of the numerator up to ``n^{-j/2}``, ``j = 0, …, 2\\,\\mathrm{order}-1``, `F[M] = Fp[2M]`;
  `Fp[1]` is the leading term), `kappa` (the cancellation indicator of the numerator),
  `kappa_den` (that of the denominator), `err` (the change with half the quadrature nodes) and
  `nG`.

All Taylor coefficients are generated by Taylor-mode differentiation directly in the canonical
coordinates of Section 5.2 of the paper, ``Y_A = X_A - CX_B``, ``Y_B = L^{-1}X_B``, in which the
tail components are uncorrelated with the conditioning components under the tilted distribution
and the latter are whitened; this is what keeps the terms from cancelling in double precision.
The computation runs in the floating-point type of `x` (`BigFloat` inside `setprecision` for
variable precision). See [`spa_cond`](@ref) for the details and the derivative interface.
"""
function conditional_tailprob(K::CGF, x::AbstractVector, A; order::Integer=3, lower::Bool=false,
                              signs=nothing, lattice::Bool=false, sinh::Symbol=:saddle, nG::Integer=64)
    _checkorder("conditional_tailprob", order)
    d = K.d
    length(x) == d || throw(ArgumentError("conditional_tailprob: x must have length $d"))
    Av = _cond_set(A, d, "conditional_tailprob")
    sg = _cond_signs("conditional_tailprob", d, Av, lower, signs)
    Kr = all(==(1), sg) ? K : reflect(K, sg)
    T = float(eltype(x))
    xr = T.(sg .* collect(x))
    maxdeg = 2order + 1
    r = _spa_cond(_kfun(Kr, maxdeg), _kwhite(Kr, maxdeg), xr, Av, order; density=false, lattice,
                  sinh, nG)
    return (; F=r.P, tau=sg .* r.tau, taut=r.taut, status=r.status,
            info=(; Fp=r.Pp, kappa=r.kappaP, kappa_den=r.kappaB, err=r.err, nG))
end

"""
    conditional_density(K::CGF, x, A; order=3, lattice=false) -> (; f, tau, taut, status, info)

Saddlepoint expansions of orders `1:order` of the conditional density ``f(x_A ∣ x_B)`` of the
components `A` of the random vector with cumulant generating function `K`, given the others,
``B``, at the point `x` (Theorem 2 of Broda and Kan, "Conditional saddlepoint expansions of
arbitrary order"): the ratio of the density expansions of ``X`` at its saddlepoint and of ``X_B``
at its own, both with the terms ``m ≤ M-1`` at order `M`, relative error ``O(n^{-M})``. The
leading term is the double saddlepoint density approximation of Barndorff-Nielsen and Cox (1979).
Unlike the distribution function, the expansion holds for saddlepoints in any orthant. With
`lattice = true` (integer-valued ``S``) the same formula is ``P(S_A = k_A ∣ S_B = k_B)``
(Theorem 3), evaluated at the uncorrected point `x`; the keyword only records this
interpretation. `tau` is the saddlepoint of `K` at `x`, `taut` that of the marginal of ``X_B``,
embedded as ``(0_A, \\tilde s_B)``; `info` holds `kappa` (the cancellation indicator of the
density expansion of ``X``) and `kappa_den` (that of ``X_B``). Both expansions are evaluated in
whitened coordinates, with the Taylor coefficients generated there directly.
"""
function conditional_density(K::CGF, x::AbstractVector, A; order::Integer=3, lattice::Bool=false)
    _checkorder("conditional_density", order)
    d = K.d
    length(x) == d || throw(ArgumentError("conditional_density: x must have length $d"))
    Av = _cond_set(A, d, "conditional_density")
    T = float(eltype(x))
    maxdeg = 2order + 1
    r = _spa_cond(_kfun(K, maxdeg), _kwhite(K, maxdeg), T.(collect(x)), Av, order; tail=false)
    return (; f=r.f, tau=r.tauf, taut=r.taut, status=r.status,
            info=(; kappa=r.kappaf, kappa_den=r.kappaB))
end
