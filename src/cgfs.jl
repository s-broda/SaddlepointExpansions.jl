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

"""
    quadratic_forms_cgf(Mats; Sigma=nothing, mu=nothing, scale=1) -> CGF

Joint cumulant generating function of the quadratic forms ``Q_i = z'A_iz``, ``i = 1, …, d``,
`Mats = [A_1, …, A_d]` (symmetric ``p × p``), in ``z ∼ N(μ, Σ)`` (default standard normal):
```math
K(t) = -\\tfrac12\\log\\det(I - 2C) + ν'C(I-2C)^{-1}ν,\\qquad
C = \\sum_i t_i\\,Σ^{1/2}A_iΣ^{1/2},\\quad ν = Σ^{-1/2}μ,
```
multiplied by `scale` (the cumulant generating function of the sum of `scale` independent copies
when `scale` is an integer). The domain is the set where ``I - 2C`` is positive definite. The
determinant and the solve are computed through a Cholesky factorization written in generic
arithmetic, so that the function can be differentiated by Taylor-mode differentiation; the cost
of one evaluation with derivatives to degree ``N`` is about ``p^3/6`` products of truncated
polynomials. For large ``p``, the closed-form derivatives of [`qf_Kmulti`](@ref) (trace
recursion) are faster.
"""
function quadratic_forms_cgf(Mats::AbstractVector; Sigma=nothing, mu=nothing, scale::Real=1)
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
    return CGF(K, d; domain=dom)
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
