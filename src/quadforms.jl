# Derivatives of the joint cumulant generating function of quadratic forms in normal variables.

"""
    qf_Kmulti(theta, Mats, L; Sinv=nothing, logdetSigma=nothing, maxidx=nothing)
        -> Dict{NTuple{p,Int},T} or nothing

Derivatives at `theta` of the joint cumulant generating function of the quadratic forms
``z'A_iz``, ``i=1,…,p`` (`Mats = [A_1, …, A_p]`, symmetric), for all multi-indices ``α`` with
``|α| ≤ L``:
```math
K(θ) = -\\tfrac12\\log\\det(I-2C),\\quad C=\\sum_iθ_iA_i,\\qquad
K_α = \\frac{2^{|α|-1}α!}{|α|}\\operatorname{tr}F_α,\\quad F_α=\\sum_i DA_iF_{α-e_i},
\\quad D=(I-2C)^{-1},
```
for ``z∼N(0,I)``. For ``z∼N(0,Σ)`` either transform the matrices, ``A↦Σ^{1/2}AΣ^{1/2}``, or pass
`Sinv` ``=Σ^{-1}`` and `logdetSigma` ``=\\log\\det Σ``; then ``K(θ)=-\\tfrac12\\{\\log\\det Σ+
\\log\\det(Σ^{-1}-2C)\\}`` and ``D=(Σ^{-1}-2C)^{-1}``.

The result is a `Dict` keyed by the multi-index as a tuple, e.g. `K[(1, 0, 2)]`; `K[(0, 0, 0)]`
is ``K(θ)``. Returns `nothing` outside the domain (``I-2C`` not positive definite).
`maxidx` (a tuple) restricts the multi-indices to ``α ≤`` `maxidx` componentwise. The traces are
formed as ``\\operatorname{tr}F_α=\\sum_{|β|=⌊|α|/2⌋}\\operatorname{tr}(F_βF_{α-β})``, so that only
the ``F_β`` with ``|β| ≤ ⌈L/2⌉`` are formed. Generic in the element type of the matrices
(e.g. `BigFloat`).
"""
function qf_Kmulti(theta::AbstractVector, Mats::AbstractVector, L::Integer;
                   Sinv=nothing, logdetSigma=nothing, maxidx=nothing)
    p = length(Mats)
    T = float(promote_type(eltype(theta), eltype(Mats[1])))
    n = size(Mats[1], 1)
    C = zeros(T, n, n)
    for i in 1:p
        C .+= T(theta[i]) .* Mats[i]
    end
    Q = Sinv === nothing ? Matrix{T}(I, n, n) - 2C : T.(Sinv) - 2C
    Ch = cholesky(Symmetric(Q); check=false)
    issuccess(Ch) || return nothing
    ldQ = 2*sum(log, diag(Ch.U))
    K0 = Sinv === nothing ? -ldQ/2 : -(T(logdetSigma) + ldQ)/2
    DA = [Ch \ T.(Mats[i]) for i in 1:p]
    cap = maxidx === nothing ? ntuple(_ -> L, p) : Tuple(maxidx)
    H = cld(L, 2)
    F = Dict{NTuple{p,Int},Matrix{T}}()
    F[ntuple(_ -> 0, p)] = Matrix{T}(I, n, n)
    for m in 1:H
        for be in compositions(m, p)
            all(be .<= cap) || continue
            X = zeros(T, n, n)
            for i in 1:p
                be[i] > 0 || continue
                bb = Base.setindex(be, be[i] - 1, i)
                X .+= m == 1 ? DA[i] : DA[i]*F[bb]
            end
            F[be] = X
        end
    end
    K = Dict{NTuple{p,Int},T}()
    K[ntuple(_ -> 0, p)] = K0
    for m in 1:L
        h = m ÷ 2
        for al in compositions(m, p)
            all(al .<= cap) || continue
            if m <= H
                tr_ = tr(F[al])
            else
                tr_ = zero(T)
                for be in compositions(h, p)
                    all(be .<= al) || continue
                    tr_ += trprod(F[be], F[al .- be])
                end
            end
            K[al] = T(2)^(m - 1)*prod(factT(T, a) for a in al)*tr_/m
        end
    end
    return K
end

# tr(X*Y) without forming the product
function trprod(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    s = zero(T)
    @inbounds for j in axes(X, 2), i in axes(X, 1)
        s += X[i, j]*Y[j, i]
    end
    return s
end

# all nonnegative integer p-tuples with sum m
function compositions(m::Integer, p::Integer)
    p == 1 && return [(m,)]
    out = NTuple{p,Int}[]
    for k in m:-1:0
        for r in compositions(m - k, p - 1)
            push!(out, (k, r...))
        end
    end
    return out
end
