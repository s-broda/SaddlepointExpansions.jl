# Cumulant generating functions as Julia functions, and their derivatives by Taylor-mode
# automatic differentiation (TaylorSeries.jl).

"""
    CGF(K, d; domain=nothing)

A cumulant generating function ``K`` of a random vector of dimension `d`, given as a Julia
function of one argument, a vector `t` of length `d` (also when `d = 1`: write `t[1]`). `K` must
be written generically: it is evaluated on vectors of `TaylorN` series to generate derivatives,
and on `BigFloat` values for variable precision, so it should use only arithmetic and the
elementary functions (`exp`, `log`, `sqrt`, powers, `sum`, matrix products with constant
matrices, ...), without type assertions or calls that force `Float64`.

Outside the domain of ``K`` the function may throw a `DomainError` (as `log` and `sqrt` of
negative arguments do) or return a non-finite value; both are understood as "outside the domain".
`domain`, an optional predicate `t -> Bool`, states the domain explicitly, which is cheaper and
more robust than relying on exceptions when the formula for ``K`` is finite outside its domain
(for instance ``-\\log(1-t)`` for ``t > 1`` is a `DomainError`, but ``1/(1-t)`` is not).

The keyword `taylor` supplies the Taylor polynomials in closed form instead of by automatic
differentiation: a function `(t, N, M) -> TaylorN` returning the Taylor polynomial of degree `N`
in `v` of ``K(t + Mv)`` (`M = nothing` for the identity; `M` may be rectangular), or `nothing`
outside the domain, built from a coefficient table with [`taylor_from_coefficients`](@ref).
The library uses it where closed forms are faster ([`quadratic_forms_cgf`](@ref), by the trace
recursion), and the transformations carry it along.

A `CGF` is callable: `K(t)` evaluates the function. See [`derivatives`](@ref),
[`saddlepoint`](@ref), [`tailprob`](@ref), [`density`](@ref), and the transformations
[`reflect`](@ref), [`marginal`](@ref), [`linear_combination`](@ref), [`sample_mean`](@ref),
[`shift`](@ref) and `+` (sum of independent vectors).

# Example
```julia
K = CGF(t -> -3*log(1 - t[1]) - 2*log(1 - t[2]) - log(1 - t[1] - t[2]), 2;
        domain = t -> t[1] < 1 && t[2] < 1 && t[1] + t[2] < 1)
```
"""
struct CGF{F,D,T}
    K::F
    d::Int
    domain::D
    taylor::T
    function CGF(K, d::Integer; domain=nothing, taylor=nothing)
        d >= 1 || throw(ArgumentError("CGF: the dimension must be a positive integer"))
        new{typeof(K),typeof(domain),typeof(taylor)}(K, Int(d), domain, taylor)
    end
end

(K::CGF)(t) = K.K(t)
Base.show(io::IO, K::CGF) = print(io, "CGF of dimension ", K.d, K.domain === nothing ? "" : " with domain predicate",
                                  K.taylor === nothing ? "" : " and closed-form Taylor polynomials")

# Taylor algebras by (number of variables, degree), created once
const _SPACES = Dict{Tuple{Int,Int},JetSpace}()
const _SPACELOCK = ReentrantLock()
function _space(k::Integer, N::Integer)
    lock(_SPACELOCK) do
        get!(_SPACES, (Int(k), Int(N))) do
            JetSpace(order=Int(N), variables=[Symbol("u", i) for i in 1:k])
        end
    end
end

_indomain(K::CGF, t) = K.domain === nothing || K.domain(t)

# Taylor polynomial of degree N in v of K(t + M v), v of dimension size(M, 2) (M = I when
# nothing); nothing when t is outside the domain. The coefficient of v^nu is K_Y^(nu)(0)/nu! for
# K_Y(v) = K(t + M v).
"""
    taylor_from_coefficients(k, N, coeffs; T=Float64) -> TaylorN

A Taylor polynomial of degree `N` in `k` variables from a table of its coefficients:
`coeffs` is a dictionary from exponent tuples ``ν`` to the coefficient of ``v^ν``
(``K^{(ν)}/ν!``), missing entries being zero. For writing closed-form `taylor` functions of a
[`CGF`](@ref).
"""
function taylor_from_coefficients(k::Integer, N::Integer, coeffs::AbstractDict; T::Type=Float64)
    sp = _space(k, max(N, 1))
    hp = HomogeneousPolynomial{T}[]
    for deg in 0:max(N, 1)
        tab = sp.coeff_table[deg+1]
        push!(hp, HomogeneousPolynomial(sp, T[get(coeffs, Tuple(e), zero(T)) for e in tab], deg))
    end
    return TaylorN(sp, hp, max(N, 1))
end

function _taylor(K::CGF, t::AbstractVector, N::Integer, M=nothing)
    T = float(eltype(t))
    _indomain(K, t) || return nothing
    if K.taylor !== nothing
        p = K.taylor(t, N, M)
        (p === nothing || !isfinite(constant_term(p))) && return nothing
        return p
    end
    k = M === nothing ? K.d : size(M, 2)
    sp = _space(k, max(N, 1))
    v = variables(T, sp)
    arg = M === nothing ? t .+ v : t .+ M*v
    p = try
        K.K(arg)
    catch err
        err isa DomainError && return nothing
        rethrow()
    end
    if !(p isa TaylorN)                     # K constant in v (e.g. a degenerate component)
        isfinite(p) || return nothing
        return TaylorN(sp, T(p), max(N, 1))
    end
    isfinite(constant_term(p)) || return nothing
    return p
end

_fact(::Type{T}, nu) where {T} = prod(factT(T, n) for n in nu; init=one(T))
_coef(::Type{T}, p::TaylorN, nu) where {T} = T(getcoeff(p, Tuple(nu)))*_fact(T, nu)

"""
    derivatives(K::CGF, t, N) -> Dict{NTuple{d,Int},T} or nothing

All derivatives ``K^{(ν)}(t)`` of orders ``|ν| ≤ N``, keyed by the multi-index as a tuple
(`D[(0, 0)]` is ``K(t)``, `D[(1, 0)]` is ``∂K/∂t_1``, ...), by Taylor-mode differentiation in the
floating-point type of `t`. `nothing` when `t` is outside the domain.
"""
function derivatives(K::CGF, t::AbstractVector, N::Integer)
    length(t) == K.d || throw(ArgumentError("derivatives: t must have length $(K.d)"))
    N >= 0 || throw(ArgumentError("derivatives: N must be nonnegative"))
    T = float(eltype(t))
    p = _taylor(K, collect(T, t), N)
    p === nothing && return nothing
    return Dict{NTuple{K.d,Int},T}(nu => _coef(T, p, nu) for nu in mi_list(K.d, N))
end
derivatives(K::CGF, t::Real, N::Integer) = derivatives(K, [t], N)

# Kfun(nu, t) for the derivative interface: the Taylor polynomial at the last t is memoized
# (degree 2 while only derivatives of order <= 2 are requested, as in the saddlepoint iteration,
# otherwise maxdeg). Not thread-safe: create one per task.
function _kfun(K::CGF, maxdeg::Integer)
    lastt = Ref{Any}(nothing); lastN = Ref(-1); lastp = Ref{Any}(nothing)
    function Kfun(nu, t)
        T = float(eltype(t))
        k = sum(nu)
        N = k <= 2 ? 2 : max(maxdeg, k)
        tt = collect(T, t)
        if lastt[] === nothing || typeof(lastt[]) !== typeof(tt) || lastt[] != tt || N > lastN[]
            lastp[] = _taylor(K, tt, N); lastt[] = tt; lastN[] = N
        end
        p = lastp[]
        p === nothing && return T(NaN)
        return _coef(T, p, nu)
    end
    return Kfun
end

# Kwhite(nu, u, M) for the derivative interface: derivatives at u of K_Y(u) = K(M u), from the
# Taylor polynomial of K(M u + M v) in v, which the expansions evaluate in whitened coordinates
# (M = L^{-T}, K'' = L L'); M may be rectangular (d x k) for the marginal terms of the
# inclusion-exclusion. Memoized at the last (u, M).
function _kwhite(K::CGF, maxdeg::Integer)
    lastu = Ref{Any}(nothing); lastM = Ref{Any}(nothing); lastp = Ref{Any}(nothing)
    function Kwhite(nu, u, M)
        T = float(promote_type(eltype(u), eltype(M)))
        uu = collect(T, u); MM = Matrix{T}(M)
        if lastu[] === nothing || typeof(lastu[]) !== typeof(uu) || lastu[] != uu || lastM[] != MM
            lastp[] = _taylor(K, MM*uu, max(maxdeg, sum(nu)), MM); lastu[] = uu; lastM[] = MM
        end
        p = lastp[]
        p === nothing && return T(NaN)
        return _coef(T, p, nu)
    end
    return Kwhite
end

# ----- transformations of cumulant generating functions

"""
    reflect(K::CGF, signs) -> CGF

Cumulant generating function of ``(s_1X_1, …, s_dX_d)`` for signs ``s_j = ±1``:
``K_s(t) = K(s ∘ t)``. `reflect(K)` negates every component. Used by [`tailprob`](@ref) for lower
and mixed tails: ``P(X_1 ≥ x_1, X_2 ≤ x_2) = P(X_1 ≥ x_1, -X_2 ≥ -x_2)``.
"""
function reflect(K::CGF, signs::AbstractVector)
    length(signs) == K.d || throw(ArgumentError("reflect: signs must have length $(K.d)"))
    all(s -> s == 1 || s == -1, signs) || throw(ArgumentError("reflect: signs must be ±1"))
    return linear_combination(K, Matrix(Diagonal(float.(collect(signs)))))
end
reflect(K::CGF) = reflect(K, fill(-1, K.d))

# a vector with the entries of s in the positions idx and zeros (of the type of s) elsewhere
function _embed(s::AbstractVector, idx, d::Integer)
    t = [zero(s[1]) for _ in 1:d]
    for (i, j) in enumerate(idx)
        t[j] = s[i]
    end
    return t
end

"""
    marginal(K::CGF, idx) -> CGF

Cumulant generating function of the components `idx` of ``X``: ``K_A(s) = K(t)`` with ``t_j = s_i``
for ``j = `` `idx[i]` and ``t_j = 0`` otherwise.
"""
function marginal(K::CGF, idx)
    ii = collect(Int, idx)
    (allunique(ii) && all(1 .<= ii .<= K.d) && !isempty(ii)) ||
        throw(ArgumentError("marginal: idx must be distinct indices in 1:$(K.d)"))
    E = zeros(K.d, length(ii))
    for (c, j) in enumerate(ii)
        E[j, c] = 1.0
    end
    return linear_combination(K, E)
end

"""
    linear_combination(K::CGF, B) -> CGF

Cumulant generating function of ``Y = B'X`` for a ``d × k`` matrix `B`: ``K_Y(u) = K(Bu)``.
(The columns of `B` are the coefficient vectors of the ``k`` linear combinations.)
"""
function linear_combination(K::CGF, B::AbstractMatrix)
    size(B, 1) == K.d || throw(ArgumentError("linear_combination: B must have $(K.d) rows"))
    Bm = Matrix(B)
    dom = K.domain === nothing ? nothing : (u -> K.domain(Bm*u))
    tf = K.taylor === nothing ? nothing :
         ((u, N, M) -> _taylor(K, Bm*u, N, M === nothing ? Bm : Bm*M))
    return CGF(u -> K.K(Bm*u), size(Bm, 2); domain=dom, taylor=tf)
end

"""
    sample_mean(K::CGF, n) -> CGF

Cumulant generating function of the mean of `n` independent copies of ``X``: ``nK(t/n)``.
"""
function sample_mean(K::CGF, n::Real)
    n > 0 || throw(ArgumentError("sample_mean: n must be positive"))
    dom = K.domain === nothing ? nothing : (t -> K.domain(t ./ n))
    tf = K.taylor === nothing ? nothing :
         ((t, N, M) -> (p = _taylor(K, t ./ n, N, (M === nothing ? Matrix(1.0I, K.d, K.d) : Matrix(M)) ./ n);
                        p === nothing ? nothing : n*p))
    return CGF(t -> n*K.K(t ./ n), K.d; domain=dom, taylor=tf)
end

"""
    shift(K::CGF, c) -> CGF

Cumulant generating function of ``X + c``: ``K(t) + c't``.
"""
function shift(K::CGF, c::AbstractVector)
    length(c) == K.d || throw(ArgumentError("shift: c must have length $(K.d)"))
    cc = collect(c)
    function tf(t, N, M)
        p = _taylor(K, t, N, M)
        p === nothing && return nothing
        Mm = M === nothing ? Matrix(1.0I, K.d, K.d) : Matrix(M)
        T = float(promote_type(eltype(t), eltype(cc)))
        v = variables(T, _space(size(Mm, 2), max(N, 1)))
        return p + dot(cc, t) + sum((cc'*Mm)[i]*v[i] for i in 1:size(Mm, 2))
    end
    return CGF(t -> K.K(t) + sum(cc .* t), K.d; domain=K.domain, taylor=(K.taylor === nothing ? nothing : tf))
end
shift(K::CGF, c::Real) = shift(K, [c])

"""
    K1 + K2

Cumulant generating function of the sum of two independent vectors with cumulant generating
functions `K1` and `K2` (same dimension).
"""
function Base.:+(K1::CGF, K2::CGF)
    K1.d == K2.d || throw(ArgumentError("+: the cumulant generating functions must have the same dimension"))
    dom = K1.domain === nothing ? K2.domain :
          K2.domain === nothing ? K1.domain : (t -> K1.domain(t) && K2.domain(t))
    tf = (K1.taylor === nothing && K2.taylor === nothing) ? nothing :
         ((t, N, M) -> (p1 = _taylor(K1, t, N, M); p2 = _taylor(K2, t, N, M);
                        (p1 === nothing || p2 === nothing) ? nothing : p1 + p2))
    return CGF(t -> K1.K(t) + K2.K(t), K1.d; domain=dom, taylor=tf)
end
