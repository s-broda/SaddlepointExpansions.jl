# Small helpers shared by the programs.

# Normal density and distribution function, generic in the floating-point type.
normpdf(x::T) where {T<:AbstractFloat} = exp(-x^2/2)/sqrt(2*T(pi))
normcdf(x::T) where {T<:AbstractFloat} = erfc(-x/sqrt(T(2)))/2

# Exact factorial in the working type (no overflow for large k).
factT(::Type{T}, k::Integer) where {T} = T(factorial(big(k)))

# h_i = (-2)^i (1/2)_i = He_{2i}(0), i = 0..n, exact integers converted to T.
function hcoef(::Type{T}, n::Integer) where {T}
    h = Vector{T}(undef, n+1)
    v = big(1)
    h[1] = T(v)
    for i in 1:n
        v *= -(2i-1)
        h[i+1] = T(v)
    end
    return h
end

# d(r,j) = [s^j] Theta(s)^r / r!, Theta(s) = sum_{k>=1} theta[k] s^k, r,j = 1..J (Section 5.1 of
# the paper); d(0,0) = 1 is implicit.
function dcoef(theta::AbstractVector{T}, J::Integer) where {T}
    d = zeros(T, J, J)
    for j in 1:J
        d[1, j] = theta[j]
    end
    for r in 2:J, j in r:J
        s = zero(T)
        for i in 1:j-r+1
            s += theta[i]*d[r-1, j-i]
        end
        d[r, j] = s/r
    end
    return d
end

# a_{i,j} = sum_{k=0}^{j} d(k,j) h_{i+k}, with d(0,0) = 1 and d(0,j) = 0 for j > 0.
function acoef(d::AbstractMatrix{T}, h::AbstractVector{T}, i::Integer, j::Integer) where {T}
    j == 0 && return h[i+1]
    s = zero(T)
    for k in 1:j
        s += d[k, j]*h[i+k+1]
    end
    return s
end

# Nodes and weights of Gauss-Hermite quadrature for the weight exp(-x^2/2)/sqrt(2 pi), and of
# Gauss-Legendre quadrature on [-1,1] (Golub-Welsch), cached by n.
const _GH = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
const _GL = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
const _QLOCK = ReentrantLock()

function gauher(n::Integer)
    lock(_QLOCK) do
        get!(_GH, n) do
            E = eigen(SymTridiagonal(zeros(n), sqrt.(Float64.(1:n-1))))
            (E.values, E.vectors[1, :].^2)
        end
    end
end

function gauleg(n::Integer)
    lock(_QLOCK) do
        get!(_GL, n) do
            k = 1:n-1
            E = eigen(SymTridiagonal(zeros(n), k./sqrt.(4.0.*k.^2 .- 1)))
            (E.values, 2 .* E.vectors[1, :].^2)
        end
    end
end

# Tolerances of the Newton iterations: "rounding level" and start of the quadratic phase.
newton_tols(::Type{Float64}) = (1e-10, 1e-7)
newton_tols(::Type{T}) where {T} = (eps(T)^T(0.7), eps(T)^T(0.45))

"""
    spa_saddle(Kfun, x, tau0=nothing) -> (tau, info)

Saddlepoint ``K'(τ)=x`` of a cumulant generating function in any dimension `d = length(x)`.
`Kfun(nu, t)` returns the derivative of order `nu` (a `d`-tuple of integers) at `t`, `NaN`
outside the domain. Newton's method with backtracking on the convex objective ``K(τ)-τ'x`` (a
step is accepted only if it stays in the domain and decreases the objective). Convergence is
measured by the Newton decrement ``δ=\\{g'K''(τ)^{-1}g\\}^{1/2}``, ``g=K'(τ)-x``, which is
invariant under linear changes of the units of the components of ``X``; the Newton step is computed
after scaling ``K''`` to unit diagonal. Once ``δ ≤ 10^{-6}`` the iteration is in its quadratic
phase: full Newton steps are taken without the objective test, which rounding errors in ``K`` would
otherwise make fail, until ``δ ≤`` `tol` (``10^{-12}``, or ``ε^{0.7}`` in other floating-point
types) or for at most three steps (twelve). Before that phase the iteration is declared stalled,
and an error is raised, only when a step no longer changes the iterate in the working arithmetic
or the backtracking finds no decrease; there is no threshold on the size of the step in the units
of ``x``. Components of ``τ`` with ``|τ_j|K''_{jj}^{1/2}`` below `tol` are set to zero, so that a
saddlepoint on a coordinate hyperplane is not reported with a spurious sign, but only if the
decrement at the rounded point is at most ``\\max(\\mathrm{tol},2δ)``. The gradient, Hessian and
decrement are recomputed at the point actually returned, and the result is accepted if that
decrement is at most ``10^{-6}`` (an error is raised otherwise). `info` holds the number of
iterations, the decrement and the residual ``K'(τ)-x`` at the returned point, `tol`, and `snapped`
(the components set to zero). All tests are made in the working precision.
"""
function spa_saddle(Kfun, x::AbstractVector, tau0=nothing)
    T = float(tau0 === nothing ? eltype(x) : promote_type(eltype(x), eltype(tau0)))
    x = T.(collect(x)); d = length(x)
    tau = tau0 === nothing ? zeros(T, d) : T.(collect(tau0))
    length(tau) == d || throw(ArgumentError("spa_saddle: tau0 must have the length of x"))
    tol, nqmax = T === Float64 ? (T(1e-12), 3) : (eps(T)^T(0.7), 12)
    z = ntuple(_ -> 0, d)
    obj(t) = Kfun(z, t) - dot(t, x)
    f = obj(tau)
    isfinite(f) || error("spa_saddle: the starting value is outside the domain of K")
    nq = 0; it = 0
    while it < 200
        it += 1
        g, _, step, delta = _newton(Kfun, tau, x)
        if delta <= 1e-6                                 # quadratic phase: full steps, no test
            tn = tau + step
            if isfinite(obj(tn))
                tau = tn; f = obj(tau)
            end
            nq += 1
            (delta <= tol || nq >= nqmax) && break
            continue
        end
        tau + step == tau &&                             # the step is below rounding level
            error("spa_saddle: the iteration stalled at rounding level with decrement $(round(Float64(delta); sigdigits=2))")
        lam = one(T)
        local tn, fn
        while true
            tn = tau + lam*step; fn = obj(tn)
            (isfinite(fn) && fn - f <= T(1e-4)*lam*dot(g, step)) && break
            lam /= 2
            (lam < 1e-14 || tn == tau) &&
                error("spa_saddle: saddlepoint iteration failed to find a descent step (decrement $(round(Float64(delta); sigdigits=2)))")
        end
        tau = tn; f = fn
    end
    # diagnostics at the returned point, after the last step and the optional rounding to zero
    g, H, _, delta = _newton(Kfun, tau, x)
    snap = (abs.(tau) .* sqrt.(abs.(diag(H))) .< tol) .& (tau .!= 0)
    if any(snap)
        ts = copy(tau); ts[snap] .= 0
        gs, _, _, ds = _newton(Kfun, ts, x)
        if ds <= max(tol, 2delta)
            tau = ts; g = gs; delta = ds
        else
            snap .= false
        end
    end
    delta <= 1e-6 ||
        error("spa_saddle: saddlepoint iteration did not converge (Newton decrement $(round(Float64(delta); sigdigits=2)))")
    return tau, (; iterations=it, decrement=Float64(delta), residual=Float64.(g), tol=Float64(tol),
                 snapped=snap)
end

# gradient, Hessian, Newton step and Newton decrement at tau; the Hessian is tested for positive
# definiteness after scaling to unit diagonal, in the working precision
function _newton(Kfun, tau::Vector{T}, x::Vector{T}) where {T}
    d = length(x)
    e(j) = ntuple(i -> i == j ? 1 : 0, d)
    g = T[Kfun(e(j), tau) for j in 1:d] .- x
    H = T[Kfun(e(i) .+ e(j), tau) for i in 1:d, j in 1:d]
    dg = diag(H)
    all(v -> isfinite(v) && v > 0, dg) ||
        error("spa_saddle: the Hessian of K is not positive definite at the current iterate")
    Ds = one(T) ./ sqrt.(dg)                          # diagonal scaling: the Newton step does
    Hs = Ds .* H .* Ds'                               # not depend on the units of the components
    (all(isfinite, Hs) && isposdef(Symmetric((Hs + Hs')/2))) ||
        error("spa_saddle: the Hessian of K is not positive definite at the current iterate")
    step = -(Ds .* (Hs \ (Ds .* g)))
    delta = sqrt(max(-dot(g, step), zero(T)))         # Newton decrement
    return g, H, step, delta
end
