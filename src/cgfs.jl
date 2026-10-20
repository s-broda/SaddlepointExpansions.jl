# A library of cumulant generating functions, written generically so that they can be
# differentiated by Taylor-mode automatic differentiation and evaluated in any precision.

"""
    normal_cgf(mu, Sigma) -> CGF

``N(μ, Σ)``: ``K(t) = μ't + t'Σt/2``. The expansions are exact at order one.
"""
function normal_cgf(mu::AbstractVector, Sigma::AbstractMatrix)
    d = length(mu)
    size(Sigma) == (d, d) || throw(ArgumentError("normal_cgf: Sigma must be $d x $d"))
    m = collect(mu); S = Matrix(Sigma)
    return CGF(t -> sum(m .* t) + sum(t .* (S*t))/2, d)
end
normal_cgf(mu::Real, sigma2::Real) = normal_cgf([mu], reshape([sigma2], 1, 1))

"""
    gamma_cgf(a, b=1) -> CGF

Gamma distribution with shape `a` and scale `b` (mean ``ab``): ``K(t) = -a\\log(1-bt)``,
``t < 1/b``. `gamma_cgf(1, b)` is the exponential distribution with mean `b`.
"""
gamma_cgf(a::Real, b::Real=1) = CGF(t -> -a*log(1 - b*t[1]), 1; domain=t -> b*t[1] < 1)

"""
    ncx2_cgf(m, lam=0) -> CGF

Noncentral chi-squared ``χ^2_m(λ)``: ``K(t) = -(m/2)\\log(1-2t) + λt/(1-2t)``, ``t < 1/2``.
"""
ncx2_cgf(m::Real, lam::Real=0) =
    CGF(t -> -m/2*log(1 - 2t[1]) + lam*t[1]/(1 - 2t[1]), 1; domain=t -> t[1] < 1//2)

"""
    exponential_sums_cgf(A; n=1, shift=nothing) -> CGF

The mean of `n` independent copies of ``A'U + ℓ``, where ``U`` is a vector of independent unit
exponential variables, `A` is ``m × d`` and `shift` ``= ℓ`` (default ``0``):
``K(t) = -n\\sum_i\\log(1 - a_i't/n) + ℓ't``, ``a_i`` the rows of `A`; domain ``a_i't < n`` for
all ``i``. With `n = 1` this is the cumulant generating function of ``A'U + ℓ`` itself; the
examples of Wang (1990) and Kolassa and Li (2010) in the paper are of this form.
"""
function exponential_sums_cgf(A::AbstractMatrix; n::Real=1, shift=nothing)
    Am = Matrix(A); d = size(Am, 2)
    n > 0 || throw(ArgumentError("exponential_sums_cgf: n must be positive"))
    l = shift === nothing ? nothing : collect(shift)
    l === nothing || length(l) == d || throw(ArgumentError("exponential_sums_cgf: shift must have length $d"))
    K = l === nothing ? (t -> -n*sum(log.(1 .- Am*t ./ n))) :
                        (t -> -n*sum(log.(1 .- Am*t ./ n)) + sum(l .* t))
    return CGF(K, d; domain=t -> all(Am*t .< n))
end

# lower Cholesky factor of a symmetric positive definite matrix, generic in the element type
# (TaylorN, BigFloat); sqrt throws a DomainError when the matrix is not positive definite
function _chol_lower(A::AbstractMatrix)
    n = size(A, 1)
    L = [zero(A[1, 1]) for _ in 1:n, _ in 1:n]
    for j in 1:n
        v = A[j, j]
        for k in 1:j-1
            v -= L[j, k]^2
        end
        L[j, j] = sqrt(v)
        for i in j+1:n
            w = A[i, j]
            for k in 1:j-1
                w -= L[i, k]*L[j, k]
            end
            L[i, j] = w/L[j, j]
        end
    end
    return L
end

# solve L L' w = b for a lower triangular L
function _chol_solve(L::AbstractMatrix, b::AbstractVector)
    n = length(b)
    z = zero(L[1, 1]*b[1])                  # the element type of the result (TaylorN, BigFloat, ...)
    y = [z for _ in 1:n]
    for i in 1:n
        v = b[i]
        for k in 1:i-1
            v -= L[i, k]*y[k]
        end
        y[i] = v/L[i, i]
    end
    w = [z for _ in 1:n]
    for i in n:-1:1
        v = y[i]
        for k in i+1:n
            v -= L[k, i]*w[k]
        end
        w[i] = v/L[i, i]
    end
    return w
end

# Taylor polynomial of degree N in v of K(t + M v) for the joint cgf of the quadratic forms w'B_j w,
# w ~ N(m, I): K(t) = -1/2 log det(I - 2C) + 1/2 m'{(I - 2C)^{-1} - I}m, C = Σ_j t_j B_j, times
# scale. Closed form by the trace recursion in the directions Dir_i = Σ_j M_ji B_j: with
# D = (I - 2C)^{-1} = L L', G_i = L' Dir_i L and y = L'm, the coefficient of v^a is
# scale 2^{|a|-1}/|a| {tr F_a + |a| y'F_a y}, F_a the symmetrized sum of the products of the G_i in
# every order; F is formed up to degree ⌈N/2⌉ and the higher traces by splitting, tr(F_g F_{a-g}).
# nothing outside the domain (the MATLAB counterpart is qf_taylor, R. Kan).
function _qf_taylor(B::Vector{<:AbstractMatrix}, m::AbstractVector, scale::Real, t::AbstractVector,
                    N::Integer, M)
    d = length(B); n = size(B[1], 1)
    T = float(promote_type(eltype(t), eltype(B[1]), eltype(m), M === nothing ? Float64 : eltype(M)))
    Mm = M === nothing ? Matrix{T}(I, d, d) : Matrix{T}(M)
    k = size(Mm, 2)
    N = max(N, 1)
    C = zeros(T, n, n)
    for j in 1:d
        C .+= T(t[j]) .* B[j]
    end
    ch = cholesky(Symmetric(Matrix{T}(I, n, n) - 2C); check=false)
    issuccess(ch) || return nothing
    R = Matrix(ch.U)                                   # I - 2C = R'R
    L = inv(R)                                         # D = L L'
    y = L'*T.(m)
    coeffs = Dict{NTuple{k,Int},T}()
    coeffs[ntuple(_ -> 0, k)] = scale*(-sum(log.(diag(R))) + (dot(y, y) - dot(m, m))/2)
    G = Vector{Matrix{T}}(undef, k)
    for i in 1:k
        Dir = zeros(T, n, n)
        for j in 1:d
            Mm[j, i] == 0 || (Dir .+= Mm[j, i] .* B[j])
        end
        Gi = L'*Dir*L
        G[i] = (Gi + Gi')/2
    end
    h = cld(N, 2)
    hasmean = any(!iszero, m)
    F = Dict{NTuple{k,Int},Matrix{T}}()
    Fy = Dict{NTuple{k,Int},Vector{T}}()
    for l in 1:N, a in compositions(l, k)
        if l <= h
            if l == 1
                Fa = G[findfirst(>(0), a)]
            else
                Fa = zeros(T, n, n)
                for i in 1:k
                    a[i] > 0 || continue
                    Fa .+= G[i]*F[Base.setindex(a, a[i] - 1, i)]
                end
                Fa = (Fa + Fa')/2
            end
            F[a] = Fa
            val = tr(Fa)
            if hasmean
                Fy[a] = Fa*y
                val += l*dot(y, Fy[a])
            end
            coeffs[a] = scale*T(2)^(l - 1)/l*val
        else
            kk = l - h
            tval = zero(T); mq = zero(T)
            for gg in compositions(kk, k)
                all(gg .<= a) || continue
                rr = a .- gg
                tval += sum(F[gg] .* F[rr])
                hasmean && (mq += dot(Fy[gg], Fy[rr]))
            end
            coeffs[a] = scale*T(2)^(l - 1)/l*(tval + l*mq)
        end
    end
    return taylor_from_coefficients(k, N, coeffs; T)
end

"""
    quadratic_forms_cgf(Mats; Sigma=nothing, mu=nothing, scale=1, closedform=true) -> CGF

Joint cumulant generating function of the quadratic forms ``Q_i = z'A_iz``, ``i = 1, …, d``,
`Mats = [A_1, …, A_d]` (symmetric ``p × p``), in ``z ∼ N(μ, Σ)`` (default standard normal):
```math
K(t) = -\\tfrac12\\log\\det(I - 2C) + ν'C(I-2C)^{-1}ν,\\qquad
C = \\sum_i t_i\\,Σ^{1/2}A_iΣ^{1/2},\\quad ν = Σ^{-1/2}μ,
```
multiplied by `scale` (the cumulant generating function of the sum of `scale` independent copies
when `scale` is an integer). The domain is the set where ``I - 2C`` is positive definite.

With `closedform = true` (the default) the Taylor polynomials, including the whitened ones, are
computed in closed form by the trace recursion of the paper in the whitened directions (the
`taylor` option of [`CGF`](@ref)), so that large ``p`` costs ``p^3`` floating-point operations per
product of matrices rather than per product of truncated polynomials. With `closedform = false`
the function is differentiated by Taylor-mode differentiation through a Cholesky factorization
written in generic arithmetic, at about ``p^3/6`` polynomial products per evaluation.
"""
function quadratic_forms_cgf(Mats::AbstractVector; Sigma=nothing, mu=nothing, scale::Real=1,
                             closedform::Bool=true)
    d = length(Mats)
    d >= 1 || throw(ArgumentError("quadratic_forms_cgf: Mats must be nonempty"))
    p = size(Mats[1], 1)
    all(M -> size(M) == (p, p), Mats) || throw(ArgumentError("quadratic_forms_cgf: the matrices must be square and of equal size"))
    if Sigma === nothing
        B = [Matrix((M + M')/2) for M in Mats]
        nu = mu === nothing ? nothing : collect(mu)
    else
        Sh = sqrt(Symmetric(Matrix(Sigma)))
        B = [Matrix(Symmetric(Sh*((M + M')/2)*Sh)) for M in Mats]
        nu = mu === nothing ? nothing : Sh \ collect(mu)
    end
    Ip = Matrix(1.0I, p, p)
    function K(t)
        C = sum(t[i] .* B[i] for i in 1:d)
        Q = Ip - 2C
        L = _chol_lower(Q)
        v = -sum(log(L[j, j]) for j in 1:p)                 # -(1/2) log det(I - 2C)
        if nu !== nothing
            Cnu = C*nu
            v += sum(Cnu .* _chol_solve(L, nu))
        end
        return scale == 1 ? v : scale*v
    end
    dom = t -> isposdef(Symmetric(Ip - 2*sum(t[i] .* B[i] for i in 1:d)))
    m0 = nu === nothing ? zeros(p) : nu
    tf = closedform ? ((t, N, M) -> _qf_taylor(B, m0, scale, t, N, M)) : nothing
    return CGF(K, d; domain=dom, taylor=tf)
end

"""
    wishart_diag_cgf(V, n) -> CGF

The diagonal of a Wishart matrix ``W ∼ W_d(n, V)``: ``K(t) = -(n/2)\\log\\det(I - 2V^{1/2}
\\mathrm{diag}(t)V^{1/2})``, the cumulant generating function of ``d`` quadratic forms
``z'e_ie_i'z`` in ``z ∼ N(0, V)``, summed over `n` independent copies ([`quadratic_forms_cgf`](@ref)
with `scale = n`).
"""
function wishart_diag_cgf(V::AbstractMatrix, n::Real)
    d = size(V, 1)
    E = [Matrix(1.0I, d, d)[:, i] * Matrix(1.0I, d, d)[i:i, :] for i in 1:d]
    return quadratic_forms_cgf(E; Sigma=V, scale=n)
end

"""
    common_component_cgf(m, lam=zeros(length(m))) -> CGF

``Q_j = X_j + X_0``, ``j = 1, …, d``, with independent ``X_i ∼ χ^2_{m_i}(λ_i)``,
`m = [m_1, …, m_d, m_0]` and `lam` likewise: ``K(t) = \\sum_j K_j(t_j) + K_0(\\sum_j t_j)``, the
``K_i`` noncentral chi-squared cumulant generating functions ([`ncx2_cgf`](@ref)). The pairs of
quadratic forms of Section 6.4 of the paper, and the trivariate example, are of this form.
"""
function common_component_cgf(m::AbstractVector, lam::AbstractVector=zeros(length(m)))
    d = length(m) - 1
    d >= 1 || throw(ArgumentError("common_component_cgf: m must have at least two entries"))
    length(lam) == d + 1 || throw(ArgumentError("common_component_cgf: lam must have the length of m"))
    mm = collect(m); ll = collect(lam)
    k1(u, mi, li) = -mi/2*log(1 - 2u) + li*u/(1 - 2u)
    K = t -> sum(k1(t[j], mm[j], ll[j]) for j in 1:d) + k1(sum(t), mm[d+1], ll[d+1])
    return CGF(K, d; domain=t -> all(t .< 1//2) && sum(t) < 1//2)
end

"""
    sinhc(x)

``\\sinh(x)/x``, equal to one at zero, for numbers and Taylor polynomials; at a Taylor polynomial
with zero constant term the series ``\\sum_m x^{2m}/(2m+1)!`` is used, so that the removable
singularity causes no `DomainError`. The cumulant generating function of the uniform
distribution on ``[-1, 1]`` is ``\\log \\mathrm{sinhc}(t)``, and the saddlepoint iteration starts at
``t = 0``, where ``\\sinh(t)/t`` would fail. See also [`sincu`](@ref) and [`uniform_cgf`](@ref).
"""
sinhc(x::Number) = iszero(x) ? one(x) : sinh(x)/x
function sinhc(x::TaylorN)
    iszero(constant_term(x)) || return sinh(x)/x
    N = TaylorSeries.order(x)
    x2 = x*x
    s = one(x); term = one(x)
    for m in 1:N÷2
        term = term*x2/((2m)*(2m + 1))
        s += term
    end
    return s
end

"""
    sincu(x)

``\\sin(x)/x`` (unnormalized; `Base.sinc` is ``\\sin(πx)/(πx)``), equal to one at zero, for
numbers and Taylor polynomials, with the removable singularity handled as in [`sinhc`](@ref).
"""
sincu(x::Number) = iszero(x) ? one(x) : sin(x)/x
function sincu(x::TaylorN)
    iszero(constant_term(x)) || return sin(x)/x
    N = TaylorSeries.order(x)
    x2 = x*x
    s = one(x); term = one(x)
    for m in 1:N÷2
        term = -term*x2/((2m)*(2m + 1))
        s += term
    end
    return s
end

"""
    uniform_cgf(a=-1, b=1) -> CGF

Uniform distribution on ``[a, b]``: ``K(t) = \\log\\mathrm{sinhc}\\{(b-a)t/2\\} + (a+b)t/2``, finite
for all ``t`` ([`sinhc`](@ref) handles ``t = 0``).
"""
function uniform_cgf(a::Real=-1, b::Real=1)
    b > a || throw(ArgumentError("uniform_cgf: b must exceed a"))
    h = (b - a)/2; m = (a + b)/2
    return CGF(t -> log(sinhc(h*t[1])) + m*t[1], 1)
end
