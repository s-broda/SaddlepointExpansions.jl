# Guide

## Cumulant generating functions

A [`CGF`](@ref) wraps a Julia function `K(t)` of a vector `t` of length ``d`` (also for
``d = 1``, where you write `t[1]`), together with the dimension and, optionally, a predicate
describing the domain:

```julia
K = CGF(t -> -3*log(1 - t[1]) - 2*log(1 - 2t[2]), 2; domain = t -> t[1] < 1 && t[2] < 1/2)
```

The function is evaluated on vectors of truncated multivariate Taylor series to generate its
derivatives, and on `BigFloat` values for variable precision. It must therefore be written
generically: arithmetic, `exp`, `log`, `sqrt`, powers, `sum`, products with constant matrices
and vectors, the trigonometric and hyperbolic functions, all work (for ``\sinh(t)/t`` and
``\sin(t)/t`` use [`sinhc`](@ref) and [`sincu`](@ref), which handle the removable singularity at
the origin); type assertions, `Float64(...)` conversions and calls into
non-generic numerical libraries do not. For determinants and inverses of matrices that depend
on `t`, write the factorization in generic arithmetic as [`quadratic_forms_cgf`](@ref) does, or
use that constructor.

Outside its domain, `K` may throw a `DomainError` (`log` and `sqrt` of a negative argument do
this automatically) or return a non-finite value; both are treated as "outside the domain", and
the saddlepoint iteration steps back. When the formula for `K` is finite outside the true domain,
as with ``1/(1-t)``, the `domain` predicate is needed. It is also cheaper than an exception.

Cumulant generating functions can be transformed: [`reflect`](@ref) (sign changes of the
components), [`marginal`](@ref), [`linear_combination`](@ref) (``Y = B'X``),
[`sample_mean`](@ref) (``nK(t/n)``), [`shift`](@ref) (``X + c``), and `K1 + K2` for the sum of
independent vectors. A library of ready-made cumulant generating functions is described in
[the library](library.md).

## Tail probabilities

[`tailprob`](@ref)`(K, x; order=M)` returns the expansions of orders ``1, …, M`` of the orthant
probability ``P(X \ge x)`` in the vector `F`, with the saddlepoint `tau`, a `status` string and
the full result of the underlying routine in `info`:

```julia
r = tailprob(K, [4.0, 1.0]; order=3)
r.F        # three approximations, of orders 1, 2 and 3
r.tau      # saddlepoint K'(τ) = x
```

The expansion of order ``M`` keeps the terms of orders ``n^{-m/2}``, ``m \le 2M-1``, of the
expansion in powers of ``n^{-1/2}``, and has relative error ``O(n^{-M})``. In practice (see the
tables of the paper) each order gains one to two digits at moderate sample sizes, and the
first-order approximation is accurate to a few percent.

*Lower and mixed tails.* `lower = true` gives ``P(X \le x)`` and `signs` a mixed corner:

```julia
tailprob(K, x; lower=true)          # P(X ≤ x)
tailprob(K, x; signs=[1, -1])       # P(X₁ ≥ x₁, X₂ ≤ x₂)
```

The components are reflected and the probability is computed as an upper orthant probability of
the reflected vector.

*Saddlepoints with negative components.* The expansion of Theorem 1 of the paper applies to
saddlepoints in the closed positive orthant, ``\tau \ge 0``, which is the case when `x` lies above
the mean in every direction that matters. When some components of ``\tau`` are negative,
`tailprob` applies the expansion through the inclusion-exclusion identity of Section 3.3,
``P(X_1 \ge x_1, X_2 \ge x_2) = P(X_2 \ge x_2) - P(X_1 < x_1, X_2 \ge x_2)``, and so on, so that
every term is an orthant probability with a saddlepoint in the closed positive orthant and its own
expansion. This preserves the relative error bound (Theorem 2); the paper explains why a direct
evaluation would not.

*Dimension one.* With a univariate `CGF`, `x` may be a scalar and the expansion is that of
Hauschildt and Robinson, ``P(X \ge x)`` computed directly for the upper tail and
``P(X \le x)`` for the lower (`lower = true`).

*Dimensions three and more.* The integrals ``I(\nu)`` of the expansion are computed by Gauss
quadrature in three dimensions and by randomized quasi-Monte Carlo in four or more; in the latter
case `info.err` is a standard error. The keywords of [`spa_mv`](@ref) (`nG`, `nGH`, `nqmc`,
`nshift`, `qtol`, ...) are passed on. The quadrature in three dimensions takes some seconds per
point at orders four and five.

*Very small probabilities.* In two dimensions the keyword `logscale = c` of
[`spa_orthant2`](@ref) returns the probability scaled by ``e^{-c}``, for probabilities below
the range of double precision. Alternatively, compute in `BigFloat` (below).

## Densities

[`density`](@ref)`(K, x; order=M)` returns the expansions of orders ``1, …, M`` of the density at
`x` (Theorem 3 of the paper; Daniels's expansion in one dimension). The expansion of order ``M``
keeps the terms ``n^{-m}``, ``m \le M-1``, and has relative error ``O(n^{-M})``. It is a finite sum
of Hermite polynomials and needs no integration, in any dimension.

In three and more dimensions the terms can cancel by many orders of magnitude at high orders.
The cancellation is monitored and the computation is repeated automatically in `BigFloat`
when needed (`refine = true`, the default; `info.refined` says whether this happened).

## Ratios

For a bivariate `K`, the joint cumulant generating function of a numerator ``X_1`` and a
positive denominator ``X_2 > 0``:

```julia
ratio_density(K, r; order=3)              # density of X₁/X₂ at r
ratio_tailprob(K, r; order=3)             # P(X₁/X₂ ≥ r);  lower=true for P(X₁/X₂ ≤ r)
```

The density is the Rice expansion of ``(2\pi i)^{-1}\int \exp\{K(s, -rs)\}\,\partial_2K(s,-rs)\,ds``
through the saddlepoint of ``K_W(s) = K(s, -rs)`` (Theorem 4 of the paper), and the tail
probability is the univariate expansion for ``W = X_1 - rX_2`` at ``0``. The derivatives of the
amplitude ``\partial_2 K`` come from the same Taylor expansion as those of ``K_W``.

*Several ratios.* With a vector `r` of length ``d``, `K` is the cumulant generating function of
``(X_1, …, X_d, Y_1, …, Y_d)``, or of ``(X_1, …, X_d, Y)`` for a common denominator, and
`ratio_density` returns the joint density of ``(X_1/Y_1, …, X_d/Y_d)`` at `r` (Theorem 4 with the
amplitude expansion of Corollary 1 of the paper), while `ratio_tailprob` returns the orthant
probability ``P(R ≥ r)``, with `lower` and `signs` as in `tailprob`. The sample autocorrelations
at lags one and two, for example, are two ratios of quadratic forms with a common denominator:

```julia
N = 50; Mc = I - ones(N, N)/N
A(j) = (diagm(j => ones(N - j)) + diagm(-j => ones(N - j)))/2
K = quadratic_forms_cgf([Mc*A(1)*Mc, Mc*A(2)*Mc, Mc])
ratio_density(K, [-1, -1]/sqrt(N); order=5).f
```

For the noncentral ``F`` distribution, for example, with ``Q_1 \sim \chi^2_m(\lambda)`` and
``Q_2 \sim \chi^2_n`` independent, ``F = (Q_1/m)/(Q_2/n)``:

```julia
K = CGF(t -> ncx2_cgf(m, λ)([t[1]]) + ncx2_cgf(n)([t[2]]), 2; domain = t -> t[1] < 1/2 && t[2] < 1/2)
c = f*m/n
ratio_tailprob(K, c; lower=true).F        # P(F ≤ f), orders 1..3
ratio_density(K, c).f .* (m/n)            # density of F at f
```

(The univariate distributions in [`ncx2_spa`](@ref), [`ncf_spa`](@ref) and [`dncf_spa`](@ref) do
this with closed-form derivatives.)

## Studentized means

For ``Q = X/\sqrt{Y}`` with ``Y > 0``, with `K` the joint cumulant generating function of
``(X, Y)``, the expansions of Bao, Broda and Kan, *Inversion formulas and saddlepoint expansions
for Studentized means*:

```julia
studentized_density(K, q; order=3)          # density of X/√Y at q, orders 1..3
studentized_tailprob(K, q; order=3)         # P(X/√Y ≥ q);  lower=true for P(X/√Y ≤ q)
studentized_quantile(K, 0.05; order=3)      # the q with P(X/√Y ≤ q) = 0.05 under the expansion
```

Unlike a ratio, ``X - q\sqrt{Y}`` is not linear in ``(X, Y)``, and the inversion is a double
integral with a kernel in ``s`` and ``t``. The saddlepoint ``(\hat s, \hat t, \hat v)`` solves
``K_s = q v``, ``K_t = v^2``, ``qs + 2tv = 0`` ([`studentized_saddlepoint`](@ref)); the density
is the bivariate amplitude expansion about it, and the distribution function is a
Hauschildt-Robinson type expansion in ``s`` after the ``t``-integral is expanded by Laplace's
method, all Taylor coefficients being generated from the Taylor polynomial of `K` at the
saddlepoint by Newton's method and composition on power series. `order = M` keeps the terms
``m ≤ M - 1`` of the density and ``m ≤ 2M - 1`` of the distribution function (relative error
``O(n^{-M})``); `info.Fp` holds every half-order. The saddlepoints for all ``q`` lie on the curve
``sK_s + 2tK_t = 0`` ([`studentized_locus`](@ref)), which is how `studentized_quantile` searches:
along the curve each ``\hat s`` gives its ``q`` and its distribution function directly, so that
no saddlepoint equation is solved for a trial ``q``.

The noncentral ``t`` distribution, for example, with ``X \sim N(\mu, 1)`` and
``Y \sim \chi^2_\nu/\nu`` independent, ``n`` pairs: ``\bar X/\sqrt{\bar Y} = T/\sqrt{n}``,
``T`` noncentral ``t`` with ``n\nu`` degrees of freedom and noncentrality ``\sqrt{n}\mu``:

```julia
K = sample_mean(CGF(t -> μ*t[1] + t[1]^2/2 - ν/2*log(1 - 2t[2]/ν), 2; domain = t -> t[2] < ν/2), n)
studentized_tailprob(K, x/sqrt(n); lower=true).F     # P(T ≤ x)
```

and the limiting Dickey-Fuller ``t``-ratio, the Studentized mean ``U/\sqrt{V}`` of two
functionals of a near-unit-root diffusion ([`dickey_fuller_cgf`](@ref)):

```julia
K = dickey_fuller_cgf(2)                                # a constant in the regression, unit root
studentized_tailprob(K, -2.862; lower=true).F           # the limiting size of the 5% test, ≈ 0.05
studentized_tailprob(dickey_fuller_cgf(2; c=-5.0), -2.862; lower=true).F   # local power at c = -5
```

The density's terms of high order cancel by many orders of magnitude where ``|\hat t|`` is
large; `studentized_density` monitors the cancellation and repeats the computation in `BigFloat`
when it exceeds ``10^6``.

## Derivatives and the saddlepoint

[`derivatives`](@ref)`(K, t, N)` returns all derivatives ``K^{(\nu)}(t)``, ``|\nu| \le N``, as a
dictionary keyed by the multi-index, and [`saddlepoint`](@ref)`(K, x)` solves ``K'(\tau) = x`` by
Newton's method with backtracking. Both are what `tailprob` and `density` use internally; the
derivatives at the saddlepoint are generated once, to the degree ``2M + 1`` needed by the order
``M``, and the whitened Taylor coefficients (see below) are generated directly rather than by
substitution.

## Closed-form Taylor polynomials

Automatic differentiation is the default, but a `CGF` can carry its Taylor polynomials in closed
form through the keyword `taylor`, a function `(t, N, M)` returning the Taylor polynomial of
degree `N` of ``K(t + Mv)`` as a `TaylorN` (built from a coefficient table with
[`taylor_from_coefficients`](@ref)), or `nothing` outside the domain. [`quadratic_forms_cgf`](@ref)
uses this with the trace recursion of the paper evaluated directly in the whitened directions,
which makes large matrices cheap, and the transformations carry the option along. Pass
`closedform = false` to differentiate the generic formula instead.

## Variable precision

Every computation runs in the floating-point type of `x`. Inside `setprecision`, with `BigFloat`
arguments, the Taylor-mode differentiation, the saddlepoint iteration and the expansions are all
done in that precision; in two and three dimensions the orthant integrals are then computed by
the exact recursion of the paper instead of quadrature:

```julia
setprecision(BigFloat, 140) do            # about 40 digits
    tailprob(K, big.([4.0, 1.0]); order=5)
end
```

## Whitening and accuracy

The terms of the multivariate expansions are evaluated for the whitened vector
``Y = L^{-1}X``, ``K''(\tau) = LL'``, in which they cancel far less than in the original
coordinates when the components are strongly correlated under the tilted distribution (Appendix B
of the paper). This needs the Taylor coefficients of ``K_Y(u) = K(L^{-T}u)``, which the package
obtains by differentiating the composed function directly. The substitution of the linear map
into the polynomial in the original coordinates, which is what the derivative interface does
when no whitened derivatives are supplied, can lose many digits at high degree; with the
Taylor-mode approach the whitened coefficients are as accurate as the original ones.

`status` and the diagnostics in `info` (the cancellation indicators `kappa`, the quadrature
checks `qerr`/`err`, the coefficient-error estimates) report the numerical quality of the
evaluation. They do not measure the asymptotic error of the expansion itself, which is what the
order controls.

## Cost

The cost is dominated by the orthant integrals, not by the differentiation: a Taylor expansion of
degree 11 of a typical cumulant generating function in three variables takes well under a
millisecond, while the quadrature for a trivariate orthant probability at order five takes
seconds. Densities are cheap in any dimension.

## Thread safety

The functions create their own caches per call and can be called from several tasks at once.
The Taylor algebras (one per dimension and degree) are created once and shared under a lock.
