# Orthant probabilities for saddlepoints of any sign: the inclusion-exclusion (17) of the paper.

"""
    spa_orthant2(Kfun, x, order; nG=200, Imethod=:auto, Kwhite=nothing, logscale=0, qtol=1e-8)
        -> (; Fup, tau, canc, kappa, status)

``P(X_1 ≥ x_1, X_2 ≥ x_2)`` of orders `1:order` by the bivariate expansion of Theorem 1, with
the inclusion-exclusion of Section 3.3 when the saddlepoint has negative components: the
full-dimensional term uses [`spa_bi`](@ref) with the reflected cumulant generating function, the
one-dimensional terms use [`spa_uni`](@ref) with their own saddlepoints. `Kfun` and the keywords
as in `spa_bi` (`Kwhite` is transformed for the negated components). With `logscale = c` every
term, the marginal and constant terms included, is scaled by ``e^{-c}`` (the marginal terms are
computed in double precision before the scaling). `canc`, `kappa`, `status`: those of the
bivariate term of the highest order (they do not measure cancellation between the terms of the
inclusion-exclusion). Lower and mixed corner
probabilities are obtained by negating components, e.g.
`Kneg = (nu, t) -> (-1)^nu[1]*Kfun(nu, [-t[1], t[2]])` at `[-x[1], x[2]]`.
"""
function spa_orthant2(Kfun, x::AbstractVector, order::Integer; nG::Integer=200,
                      Imethod::Symbol=:auto, Kwhite=nothing, logscale::Real=0, qtol::Real=1e-8)
    T = float(eltype(x))
    x = T.(collect(x))
    r = spa_bi(Kfun, x, order; nG, Imethod, Kwhite, logscale, qtol)
    tau = r.tau
    canc = r.canc[end]; kappa = r.kappa[end]; status = r.status
    all(tau .>= 0) && return (; Fup=r.Fup, tau, canc, kappa, status)
    Nn = findall(tau .< 0)
    Fup = zeros(T, order)
    for mask in 0:2^length(Nn)-1
        Tset = [Nn[i] for i in eachindex(Nn) if (mask >> (i - 1)) & 1 == 1]
        sgn = (-1)^length(Tset)
        keep = setdiff(1:2, setdiff(Nn, Tset))
        flip = ones(Int, 2); flip[Tset] .= -1
        if length(keep) == 2
            Kf = (nu, t) -> flip[1]^nu[1]*flip[2]^nu[2]*Kfun(nu, flip .* t)
            Kw = Kwhite === nothing ? nothing :
                 (nu, u, M) -> Kwhite(nu, u, Diagonal(flip)*M)     # K_f(M u) = K(F M u)
            r2 = spa_bi(Kf, flip .* x, order; tau0=abs.(tau), nG, Imethod, Kwhite=Kw, logscale, qtol)
            canc = r2.canc[end]; kappa = r2.kappa[end]; status = r2.status
            Fup .+= sgn .* r2.Fup
        elseif length(keep) == 1
            j = keep[1]; f = flip[j]
            K1 = (k, s) -> f^k*Kfun(j == 1 ? (k, 0) : (0, k), j == 1 ? [f*s, zero(s)] : [zero(s), f*s])
            F1 = uni_upper(K1, f*x[j], order)
            Fup .+= sgn .* sign.(F1) .* exp.(log.(abs.(F1)) .- logscale)
        else
            Fup .+= sgn*exp(-T(logscale))
        end
    end
    return (; Fup, tau, canc, kappa, status)
end

# P[X >= xj] for the univariate cgf K1(k, s) (NaN outside the domain), by spa_uni; the
# saddlepoint by spa_saddle in one dimension
function uni_upper(K1, xj::T, order::Integer) where {T}
    N = 2order + 1
    s, _ = spa_saddle((nu, t) -> K1(nu[1], t[1]), [xj], [zero(T)])
    s = s[1]
    Kd = [K1(k, s) for k in 0:N]
    return spa_uni(Kd, s, xj, order).upp
end
