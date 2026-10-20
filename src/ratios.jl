# The joint density of several ratios (Theorem 4 and Corollary 1 of the paper) from a CGF: all the
# Taylor coefficients, of K_W and of the amplitude functions, come from one Taylor polynomial of
# K(s, -r∘s + τ) in (u, τ), evaluated in the whitened coordinates s = ŝ + L^{-T}u.

# the cumulant generating function of (X_1..X_d, Y_1..Y_d) from one of dimension 2d or, for a
# common denominator, d + 1
function _ratio_cgf2(K::CGF, d::Integer, name::AbstractString)
    K.d == 2d && return K
    if K.d == d + 1
        B = zeros(d + 1, 2d)
        for j in 1:d
            B[j, j] = 1.0
            B[d+1, d+j] = 1.0
        end
        return linear_combination(K, B)
    end
    throw(ArgumentError("$name: K must have dimension 2d (numerators, denominators) or d + 1 (common denominator), d = $d"))
end

# all partitions of 1:d, each a vector of block bit masks
function _setparts(d::Integer)
    parts = [[1]]
    for i in 2:d
        new = Vector{Vector{Int}}()
        b = 1 << (i - 1)
        for bl in parts
            for j in eachindex(bl)
                q = copy(bl); q[j] += b; push!(new, q)
            end
            push!(new, vcat(bl, b))
        end
        parts = new
    end
    return parts
end

_dfact(n::Integer) = n <= 0 ? 1 : prod(n:-2:1)

"""
    _ratio_mv(K2, r, order) -> (f, info)

Saddlepoint expansion of orders `1:order` of the joint density at `r` of ``R_j = X_j/Y_j``,
``j = 1..d``, from the cumulant generating function `K2` of ``(X_1..X_d, Y_1..Y_d)`` (``Y_j > 0``):
with ``W = X - r∘Y``, ``K_W(s) = K(s, -r∘s)``, ``K_W'(\\hat s) = 0`` and ``K_W'' = LL'``,
``f(r) = (2πi)^{-d}∫e^{K_W(s)}g(s)ds`` with ``g = \\sum_π\\prod_{B∈π}K_B(s, -r∘s)``, the sum over
the partitions of ``\\{1..d\\}`` and ``K_B`` the mixed derivative of ``K`` in the components of
``t`` in ``B``; ``g = g_0 + … + g_{d-1}`` with ``g_k`` the partitions into ``d - k`` blocks, and
``f ≈ e^{K_W(\\hat s)}(2π)^{-d/2}|K_W''|^{-1/2}\\sum_{m<M}a_m``, ``a_m = \\sum_k b_{m-k}(g_k)``,
``b_m(h) = \\sum_r\\sum_{|ν|+|μ|=2m+2r}c_{r,ν}γ_μ\\mathrm{He}_{ν+μ}(0)``, in the whitened
coordinates (Corollary 1 of the paper).
"""
function _ratio_mv(K2::CGF, r::AbstractVector, order::Integer)
    d = length(r); M = order
    T = float(eltype(r))
    rr = T.(collect(r))
    B = vcat(Matrix{T}(I, d, d), -Diagonal(rr))
    KW = linear_combination(K2, Matrix(B))
    s, sinfo = spa_saddle(_kfun(KW, 2), zeros(T, d), nothing)
    P2 = _taylor(KW, s, 2)
    P2 === nothing && error("ratio_density: the saddlepoint is outside the domain of K")
    e(i, j) = ntuple(l -> (l == i) + (l == j), d)
    H = [_coef(T, P2, e(i, j)) for i in 1:d, j in 1:d]
    ch = cholesky(Symmetric((H + H')/2); check=false)
    issuccess(ch) || error("ratio_density: K_W'' is not positive definite at the saddlepoint")
    L = Matrix(ch.L); Lit = Matrix(inv(L)')
    N = max(2M, 2M - 2 + d)
    Mmap = [Lit zeros(T, d, d); -Diagonal(rr)*Lit Matrix{T}(I, d, d)]
    P = _taylor(K2, vcat(s, -rr .* s), N, Mmap)
    P === nothing && error("ratio_density: the saddlepoint is outside the domain of K")
    D = max(6(M - 1), 2M)
    zd = ntuple(_ -> 0, d)
    # Theta: the terms of degree 3..2M of K_W(ŝ + L^{-T}u)
    theta = Dict{NTuple{d,Int},T}()
    for u in mi_list(d, min(2M, N))
        3 <= sum(u) || continue
        theta[u] = _coef_raw(T, P, (u..., zd...))
    end
    Theta = taylor_from_coefficients(d, D, theta; T)
    K0 = _coef_raw(T, P, ntuple(_ -> 0, 2d))
    # K_B(ŝ + L^{-T}u) for every nonempty subset B (bit mask), degree ≤ 2(M-1) in u
    G = Vector{TaylorN{T}}(undef, 2^d - 1)
    for mask in 1:2^d-1
        tv = ntuple(j -> (mask >> (j - 1)) & 1, d)
        cf = Dict{NTuple{d,Int},T}()
        for u in mi_list(d, min(2(M - 1), N - sum(tv)))
            cf[u] = _coef_raw(T, P, (u..., tv...))
        end
        G[mask] = taylor_from_coefficients(d, D, cf; T)
    end
    # g_k: the partitions into d - k blocks, truncated at degree 2(M-1)
    g = [taylor_from_coefficients(d, D, Dict{NTuple{d,Int},T}(); T) for _ in 1:d]
    for bl in _setparts(d)
        q = G[bl[1]]
        for j in 2:length(bl)
            q = q*G[bl[j]]
        end
        k = d - length(bl)
        g[k+1] = g[k+1] + q
    end
    g = [_truncate(h, 2(M - 1), d, D, T) for h in g]
    # He_ν(0) = ∏_j He_{ν_j}(0), and the terms a_m
    he(nu) = prod(iseven(n) ? (-1)^(n ÷ 2)*_dfact(n - 1) : 0 for n in nu; init=1)
    R = 2(M - 1)
    Pr = Vector{TaylorN{T}}(undef, R + 1)
    Pr[1] = taylor_from_coefficients(d, D, Dict(zd => one(T)); T)
    for q in 1:R
        Pr[q+1] = Pr[q]*Theta/q
    end
    a = zeros(T, M)
    exps = mi_list(d, D)
    for k in 0:min(M - 1, d - 1)
        h = g[k+1]
        for q in 0:R
            Q = Pr[q+1]*h
            for m in cld(q, 2):M-1-k
                tot = 2m + 2q
                for nu in exps
                    sum(nu) == tot || continue
                    a[m+k+1] += getcoeff(Q, nu)*he(nu)
                end
            end
        end
    end
    logC = K0 - T(d)/2*log(2*T(pi)) - sum(log.(diag(L)))
    f = exp(logC) .* cumsum(a)
    return f, (; shat=s, a, logC, saddle=sinfo)
end

# coefficient (not the derivative) of v^nu of a TaylorN
_coef_raw(::Type{T}, p::TaylorN, nu) where {T} = T(getcoeff(p, Tuple(nu)))

# the terms of degree ≤ deg of h, as a TaylorN of degree D in d variables
function _truncate(h::TaylorN, deg::Integer, d::Integer, D::Integer, ::Type{T}) where {T}
    cf = Dict{NTuple{d,Int},T}()
    for u in mi_list(d, min(deg, D))
        cf[u] = T(getcoeff(h, u))
    end
    return taylor_from_coefficients(d, D, cf; T)
end
