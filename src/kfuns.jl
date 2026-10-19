# Derivative-interface cumulant generating functions (Kfun(nu, t)) of the examples of the paper,
# with closed-form derivatives, and the first-order approximations of Wang (1990) and of Kolassa
# and Li (2010) for comparisons.

"""
    lin_exp_Kfun(nu, t, A, n, lin=zeros(d))

Derivative of order `nu` (a `d`-tuple) at `t` of the cumulant generating function of the mean of
`n` independent copies of ``A'U+\\ell``, ``U`` a vector of independent unit exponential variables:
``K(t)=n\\sum_i[-\\log(1-a_i't/n)]+\\ell't``, ``a_i`` the rows of `A` (``m×d``). `NaN` outside
the domain. Closed-form counterpart of [`exponential_sums_cgf`](@ref).
"""
function lin_exp_Kfun(nu, t::AbstractVector, A::AbstractMatrix, n::Real, lin=nothing)
    T = float(promote_type(eltype(t), typeof(n)))
    k = sum(nu)
    z = 1 .- A*t ./ n
    any(z .<= 0) && return T(NaN)
    if k == 0
        v = -n*sum(log.(z))
        lin === nothing || (v += dot(lin, t))
        return T(v)
    end
    v = zero(T)
    for i in axes(A, 1)
        an = prod(A[i, j]^nu[j] for j in eachindex(nu))
        iszero(an) && continue
        v += n*factT(T, k - 1)*an/T(n)^k/z[i]^k
    end
    if k == 1 && lin !== nothing
        v += lin[findfirst(>(0), collect(nu))]
    end
    return v
end

"""
    wishdiag_kfun(V, n; Lmax=16) -> Kfun

Function `Kfun(nu, t)` returning the derivative of order `nu = (nu1, nu2, nu3)` at `t` of the
cumulant generating function of the diagonal of a Wishart ``W_3(n,V)`` matrix,
``K(t)=-(n/2)\\log\\det(I-2V^{1/2}\\mathrm{diag}(t)V^{1/2})``, the cgf of three quadratic forms:
``D_ν=n2^{|ν|-1}ν!\\,\\mathrm{tr}(F_ν)/|ν|``, ``F_ν=\\sum_i(DA_i)F_{ν-e_i}``,
``A_i=V^{1/2}e_ie_i'V^{1/2}``, ``D=(I-2\\sum_it_iA_i)^{-1}``. `NaN` outside the domain.
Memoized: all derivatives with ``|ν| ≤ L`` are computed at the first request at a new `t`
(``L=2`` when only derivatives of order at most two are requested, otherwise `Lmax` or ``|ν|``).
Closed-form counterpart of [`wishart_diag_cgf`](@ref).
"""
function wishdiag_kfun(V::AbstractMatrix, n::Real; Lmax::Integer=16)
    lastt = Ref{Any}(nothing); lastL = Ref(-1); lastD = Ref{Any}(nothing)
    function Kfun(nu, t)
        k = sum(nu)
        L = k <= 2 ? 2 : max(Lmax, k)
        if lastt[] === nothing || lastt[] != t || L > lastL[]
            lastD[] = _wishdiag_all(collect(t), L, V, n); lastt[] = copy(collect(t)); lastL[] = L
        end
        lastD[] === nothing && return float(eltype(t))(NaN)
        return lastD[][nu[1]+1, nu[2]+1, nu[3]+1]
    end
    return Kfun
end

function _wishdiag_all(t::AbstractVector, L::Integer, V::AbstractMatrix, n::Real)
    T = float(eltype(t)); d = 3
    Vh = sqrt(Symmetric(T.(V)))
    A = [Vh[:, i]*Vh[i:i, :] for i in 1:d]
    C = sum(t[i] .* A[i] for i in 1:d)
    Q = Matrix{T}(LinearAlgebra.I, d, d) .- 2 .* C
    ch = cholesky(Symmetric((Q + Q')/2); check=false)
    issuccess(ch) || return nothing
    Dm = inv(Q)
    DA = [Dm*A[i] for i in 1:d]
    D = zeros(T, L + 1, L + 1, L + 1)
    F = Dict{NTuple{3,Int},Matrix{T}}()
    D[1, 1, 1] = -n*sum(log.(diag(ch.U)))
    F[(0, 0, 0)] = Matrix{T}(LinearAlgebra.I, d, d)
    for k in 1:L, a1 in k:-1:0, a2 in k-a1:-1:0
        nu = (a1, a2, k - a1 - a2)
        Fn = zeros(T, d, d)
        for i in 1:3
            nu[i] > 0 || continue
            Fn .+= DA[i]*F[Base.setindex(nu, nu[i] - 1, i)]
        end
        F[nu] = Fn
        D[nu[1]+1, nu[2]+1, nu[3]+1] = n*T(2)^(k - 1)*prod(factT(T, v) for v in nu)*tr(Fn)/k
    end
    return D
end

"""
    kl_bivar(Kfun, x; ord=(1, 2), hrel=1e-3) -> (P, part, info)

First-order bivariate tail approximation of Kolassa and Li (2010), ``P(X_1 ≥ x_1, X_2 ≥ x_2)``,
as written out for ``d=2`` in Li (2009, Chapter 3), with ``n=1`` (pass the cgf of the vector
itself). `ord = (2, 1)` reverses the order of the coordinates in the nested signed roots.
`P = sum(part)`: main quadratic term ``C\\bar P(t;G)``, its cubic correction (third-degree terms
of ``g``, Kolassa 2003), ``I^{1}``, ``I^{2}``. `P` is `NaN` when a needed value of the cgf is
outside its domain. The second derivative ``v''`` of the nested signed root is computed by central
differences with step `hrel*max(1,|τ_1|)`. The formula has removable singularities on the lines
``τ_1=0`` and ``x_2=E(X_2)``; this routine evaluates it directly in double precision, and near
those lines the result is dominated by rounding (Table 6 of the paper).
"""
function kl_bivar(Kfun, x::AbstractVector; ord=(1, 2), hrel::Real=1e-3)
    if Tuple(ord) == (2, 1)
        K = (nu, t) -> Kfun((nu[2], nu[1]), [t[2], t[1]]); x = [x[2], x[1]]
    else
        K = Kfun; x = collect(Float64, x)
    end
    part = fill(NaN, 4)
    nan = (NaN, part, (;))
    tau = _kl_saddle2(K, x)
    K0 = K((0, 0), tau) - dot(tau, x)
    t20 = _kl_inner(K, 0.0, x[2], tau[2])
    isnan(t20) && return nan
    f0 = K((0, 0), [0.0, t20]) - t20*x[2]
    w2h = sign(t20)*sqrt(max(-2*f0, 0.0))
    w1h = sign(tau[1])*sqrt(max(-2(K0 - f0), 0.0))
    Kt10 = K((0, 0), [tau[1], 0.0])
    isnan(Kt10) && return nan
    w2c = w2h - sign(tau[2])*sqrt(max(2(Kt10 - tau[1]*x[1] - K0), 0.0))
    H = [K((2, 0), tau) K((1, 1), tau); K((1, 1), tau) K((0, 2), tau)]
    fpp = H[1, 1] - H[1, 2]^2/H[2, 2]
    prof(s) = (g = _kl_inner(K, s, x[2], tau[2]); K((0, 0), [s, g]) - s*x[1] - g*x[2])
    w1f(s) = w1h + sign(s - tau[1])*sqrt(max(2(prof(s) - K0), 0.0))
    Wf(s) = w2h + sign(-_kl_inner(K, s, x[2], tau[2]))*sqrt(max(2(K((0, 0), [s, 0.0]) - s*x[1] - prof(s)), 0.0))
    vp = (K((1, 0), [tau[1], 0.0]) - x[1])/((w2c - w2h)*sqrt(fpp))
    h = hrel*max(1, abs(tau[1]))
    s = tau[1] .+ h .* (-2:2)
    w1v = w1f.(s); Wv = Wf.(s)
    d1w = (w1v[4] - w1v[2])/(2h); d2w = (w1v[4] - 2w1v[3] + w1v[2])/h^2
    d1W = (Wv[4] - Wv[2])/(2h);   d2W = (Wv[4] - 2Wv[3] + Wv[2])/h^2
    vpp = (d2W*d1w - d1W*d2w)/d1w^3
    info = (; tau, t20, w1h, w2h, w2c, vp, vp_fd=d1W/d1w, vpp)
    all(isfinite, (vp, vpp, w1h, w2h, w2c)) || return (NaN, part, info)
    G = [1+vp^2 vp; vp 1]
    uh = [w1h, w2h - w2c]
    tt = G*uh
    C = exp(-0.5(w1h^2 + w2h^2) + 0.5dot(uh, G*uh))
    Pbar = orthant2(tt, G)                     # P(Z <= -tt) = P(Z >= tt), Z ~ N(0, G)
    I = _kl_Inu3(G, tt, uh)
    part[1] = C*Pbar
    part[2] = C*(3vp*vpp/6*I[4, 1] + vpp/2*I[3, 2])
    hh = 1/(tau[1]*sqrt(fpp)) - 1/w1h
    a1 = 1 + vp^2
    part[3] = C*hh/sqrt(a1)*normpdf(tt[1]/sqrt(a1))*normcdf(-(w2h - w2c)/sqrt(a1))
    K22 = K((0, 2), [0.0, t20])
    part[4] = (1/(t20*sqrt(K22)) - 1/w2h)*normcdf(-w1h)*normpdf(w2h)
    return (sum(part), part, info)
end

"""
    wang_bivar(Kfun, x; ord=(1, 2)) -> (F, part, info)

First-order bivariate approximation of Wang (1990, Theorem 1) to the distribution function
``F(x_1, x_2) = P(X_1 ≤ x_1, X_2 ≤ x_2)``, for the cumulant generating function of the vector
itself (``n`` absorbed). The inner inversion integral, over the first coordinate, is approximated by
the Lugannani-Rice formula and the outer one by Watson's lemma, so that the formula depends on the
order of the coordinates; `ord = (2, 1)` reverses it. Wang recommends the order with the smaller
signed root `info.wu0`. `F = sum(part)` with the four terms of his equation (27): the bivariate
normal distribution function (17), and the corrections (18), (25) and (26). The formula needs the
cumulant generating function at ``(0, τ_2)``, the saddlepoint with its first component replaced by
zero, and `F` is `NaN` when that point is outside the domain or when a component of the saddlepoint
is zero (where Wang takes limits). The term (18) contains ``φ(v̄_0)`` with
``v̄_0^2 = -2{K(0, τ_2) - τ_2 x_2}`` (his Lemma 4), which can be negative; it is evaluated here through
``exp{K(0, τ_2) - τ_2 x_2}/√(2π)``, which is what his derivation produces and equals ``φ(v̄_0)`` when
``v̄_0^2 ≥ 0``, and `info.vbar2` reports ``v̄_0^2`` so that the user can tell (Table 7 of the paper).
Checked against the values printed in Wang (1990, Table 1).
"""
function wang_bivar(Kfun, x::AbstractVector; ord=(1, 2))
    if Tuple(ord) == (2, 1)
        K = (nu, t) -> Kfun((nu[2], nu[1]), [t[2], t[1]]); x = [x[2], x[1]]
    else
        K = Kfun; x = collect(Float64, x)
    end
    part = fill(NaN, 4)
    nan = (NaN, part, (; tau=fill(NaN, 2), wu0=NaN, v0=NaN, vbar2=NaN))
    tau = _kl_saddle2(K, x); t0, u0 = tau
    all(isfinite, tau) || return nan
    Ksw = (nu, t) -> K((nu[2], nu[1]), [t[2], t[1]])
    th1 = _kl_inner(Ksw, 0.0, x[1], t0)                  # marginal saddlepoint of X_1: K_1(th1, 0) = x_1
    isnan(th1) && return nan
    l0 = K((0, 0), [th1, 0.0]) - th1*x[1]                 # l(0)
    lu0 = K((0, 0), tau) - dot(tau, x)                    # l(u0) = K(tau) - tau'x
    K0u0 = K((0, 0), [0.0, u0])                           # K(0, u0)
    isfinite(K0u0) || return nan
    x1 = sign(th1)*sqrt(max(-2l0, 0.0))                  # signed root of the marginal of X_1
    wu0 = sign(t0)*sqrt(max(2(K0u0 - lu0 - u0*x[2]), 0.0))  # signed root of the inner integral at u0
    v0 = sign(u0)*sqrt(max(2(l0 - lu0), 0.0))             # signed root of the outer integral
    b = (wu0 - x1)/v0
    y1 = (v0 - b*x1)/sqrt(1 + b^2); rho1 = -b/sqrt(1 + b^2)
    Ktt = K((2, 0), tau); Ktu = K((1, 1), tau); Kuu = K((0, 2), tau)
    G = sqrt(Kuu - Ktu^2/Ktt)
    gx = 1/wu0 - 1/(t0*sqrt(Ktt)); gy = 1/v0 - 1/(u0*G)
    vbar2 = -2(K0u0 - u0*x[2])                            # vbar0^2 of Wang's Lemma 4
    info = (; tau, th1, x1, wu0, v0, b, y1, rho1, G, vbar2)
    all(isfinite, (x1, wu0, v0, b, y1, rho1, G, gx, gy)) && abs(rho1) < 1 || return (NaN, part, info)
    I11 = try orthant2([-x1, -y1], [1.0 rho1; rho1 1.0]) catch; return (NaN, part, info) end
    part[1] = I11                                         # P(Z_1 <= x1, Z_2 <= y1), corr. rho1
    part[2] = normcdf(wu0)*exp(K0u0 - u0*x[2])/sqrt(2pi)*gy  # (18): phi(vbar0), vbar0^2 = -2{K(0,u0) - u0 x2}
    part[3] = normcdf(v0)*normpdf(x1)*gx                  # (25)
    part[4] = normpdf(x1)*normpdf(v0)*gx*gy               # (26)
    return (sum(part), part, info)
end

# solve K_2(s, g) = x2 for g by Newton with step halving (NaN if it fails)
function _kl_inner(K, s, x2, g0)
    g = g0; isfinite(K((0, 0), [s, g])) || (g = 0.0)
    for it in 1:200
        f1 = K((0, 1), [s, g]) - x2; f2 = K((0, 2), [s, g])
        st = -f1/f2; lam = 1.0
        while !isfinite(K((0, 0), [s, g + lam*st])) && lam > 1e-12
            lam /= 2
        end
        g += lam*st
        abs(lam*st) < 1e-14*(1 + abs(g)) && break
    end
    (!isfinite(K((0, 0), [s, g])) || abs(K((0, 1), [s, g]) - x2) > 1e-8*(1 + abs(x2))) && return NaN
    return g
end

# saddlepoint by Newton with backtracking, polished by full Newton steps on the gradient (which
# the second differences of the signed roots centred at tau need)
function _kl_saddle2(K, x)
    tau = [0.0, 0.0]
    obj(t) = K((0, 0), t) - dot(t, x)
    f = obj(tau)
    grad(t) = [K((1, 0), t), K((0, 1), t)] .- x
    hess(t) = [K((2, 0), t) K((1, 1), t); K((1, 1), t) K((0, 2), t)]
    for it in 1:200
        g = grad(tau); st = -(hess(tau) \ g)
        norm(st) < 1e-14*(1 + norm(tau)) && break
        lam = 1.0; tn = tau; fn = f
        while true
            tn = tau + lam*st; fn = obj(tn)
            (isfinite(fn) && fn <= f + 1e-4*lam*dot(g, st)) && break
            lam /= 2; lam < 1e-14 && break
        end
        if norm(lam*st) < 1e-13*(1 + norm(tau))
            tau = tn; break
        end
        tau = tn; f = fn
    end
    for it in 1:4
        tn = tau - hess(tau) \ grad(tau)
        isfinite(K((0, 0), tn)) || break
        tau = tn
    end
    return tau
end

# I(nu) = I(S,-1,nu,t,sh), |nu| <= 3, from the binomial sum and the derivatives J(r) of Pbar(t;S)
function _kl_Inu3(S::Matrix{Float64}, t::Vector{Float64}, sh::Vector{Float64})
    L = 3; Si = inv(S)
    He = hermite2(L, Si, t)
    phi2 = exp(-dot(t, Si*t)/2)/(2pi*sqrt(det(S)))
    J = zeros(L + 1, L + 1)
    for a in 0:L-1, b in 0:L-1-a
        J[a+2, b+2] = phi2*He[a+1, b+1]
    end
    for i in 1:2
        j = 3 - i; Sjj = S[j, j]; Sij = S[i, j]; cv = S[i, i] - Sij^2/Sjj
        g = normpdf(t[j]/sqrt(Sjj))/sqrt(Sjj)*normcdf(-(t[i] - Sij/Sjj*t[j])/sqrt(cv))
        Jr = zeros(L); Jr[1] = g
        for rj in 0:L-2
            Jri = i == 1 ? J[2, rj+2] : J[rj+2, 2]
            v = t[j]/Sjj*Jr[rj+1] - Sij/Sjj*Jri
            rj > 0 && (v -= rj/Sjj*Jr[rj])
            Jr[rj+2] = v
        end
        if i == 1
            J[1, 2:L+1] = Jr
        else
            J[2:L+1, 1] = Jr
        end
    end
    J[1, 1] = orthant2(t, S)
    I = zeros(L + 1, L + 1)
    for k in 0:L, n1 in 0:k
        n2 = k - n1
        w1 = [binomial(n1, m)*(-sh[1])^m for m in 0:n1]
        w2 = [binomial(n2, m)*(-sh[2])^m for m in 0:n2]
        I[n1+1, n2+1] = dot(w1, J[n1+1:-1:1, n2+1:-1:1]*w2)
    end
    return I
end
