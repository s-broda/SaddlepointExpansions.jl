# Background: the expansions

This page summarizes what the package computes; the paper gives the derivations, the error
bounds and the numerical evidence.

## Setting

``X`` is a random vector of dimension ``d`` with cumulant generating function
``K(t) = \log E\,e^{t'X}``, finite on an open set containing the origin. The large parameter
``n`` (a sample size, degrees of freedom, ...) is absorbed in ``K``: for the mean of ``n``
observations, ``K_n(t) = nK(t/n)`` ([`sample_mean`](@ref)). The saddlepoint ``\tau`` at a point
``x`` solves ``K'(\tau) = x``, and ``K''(\tau)`` is the covariance matrix of the exponentially
tilted distribution.

## Densities (Theorem 3)

```math
f(x) = \frac{e^{K(\tau) - \tau'x}}{(2\pi)^{d/2}\,|K''(\tau)|^{1/2}}
\Bigl\{1 + \sum_{m \ge 1} n^{-m} b_m(\tau)\Bigr\},
```

where ``b_m`` is a finite combination of the Taylor coefficients of ``K`` at ``\tau`` of orders
three to ``2m + 2`` and of multivariate Hermite polynomials at zero. The expansion of order ``M``
keeps ``m \le M - 1`` and has relative error ``O(n^{-M})``. In one dimension this is Daniels's
expansion.

## Orthant probabilities (Theorem 1)

```math
P(X \ge x) = e^{K(\tau) - \tau'x + \tau'K''(\tau)\tau/2}
\sum_{m \ge 0} n^{-m/2} \sum_{\nu} a_{m,\nu}(\tau)\, I(\nu),
```

with coefficients ``a_{m,\nu}`` built from the same Taylor coefficients and integrals
``I(\nu) = \int_{w \ge 0} He_\nu(w; K'')\,\phi(w + K''\tau; K'')\,dw`` of Hermite polynomials
against a normal density over the orthant. The expansion of order ``M`` keeps ``m \le 2M - 1``
and has relative error ``O(n^{-M})``, uniformly for saddlepoints in compact subsets of the
closed positive orthant ``\tau \ge 0``, so including points on the boundary where some
components of the saddlepoint vanish. The term ``m = 0`` is the approximation of Kolassa (2003).
In one dimension the expansion is that of Hauschildt and Robinson, whose first-order term has
the same accuracy as the Lugannani-Rice formula.

The integrals ``I(\nu)`` are evaluated in whitened coordinates (Appendix B of the paper): for
``Y = L^{-1}X``, ``K'' = LL'``, the Hermite polynomials factor into univariate ones and the
terms cancel far less. In two dimensions ``I(\nu)`` is computed by Gauss quadrature (or by the
exact recursion of Lemma 2 in variable precision), in three by Gauss quadrature on a conditioned
grid, and in four or more by randomized quasi-Monte Carlo after Genz's sequential conditioning.

## Saddlepoints with negative components (Section 3.3, Theorem 2)

When ``\tau`` has negative components, the probability is written by inclusion-exclusion as a
signed sum of orthant probabilities of lower-dimensional marginals and of reflected vectors, each
with a saddlepoint in its closed positive orthant, and each is expanded. Theorem 2 of the paper
shows that this preserves the relative error bound under mild conditions; the paper also shows
that the direct evaluation of the formula at such a saddlepoint has the wrong exponential order.

## Ratios (Theorem 4)

For ``R = X_1/X_2`` with ``X_2 > 0``, ``P(R \ge r) = P(X_1 - rX_2 \ge 0)`` is a univariate tail
probability with cumulant generating function ``K_W(s) = K(s, -rs)``, and the density is the
Rice expansion of ``(2\pi i)^{-1}\int e^{K_W(s)}\,\partial_2K(s, -rs)\,ds`` through the
saddlepoint ``K_W'(\hat s) = 0``.

## Accuracy in practice

In the examples of the paper (means of five to ten observations of exponential variables, pairs
of quadratic forms in normal variables with ten to sixty degrees of freedom, the diagonal of a
Wishart matrix with five degrees of freedom) the first-order approximations have relative errors
of one to thirty percent, order two reduces them to ``10^{-3}`` to ``10^{-4}``, and order five to
``10^{-6}`` to ``10^{-9}``. Each doubling of ``n`` divides the error of order ``M`` by about
``2^M``.

## References

- Broda, S. A. and Kan, R. *Uniform saddlepoint expansions for multivariate tail probabilities.*
- Daniels, H. E. (1954). Saddlepoint approximations in statistics. *Ann. Math. Statist.* 25, 631-650.
- Daniels, H. E. (1987). Tail probability approximations. *Int. Statist. Rev.* 55, 37-48.
- Hauschildt, R. and Robinson, J. (1976). *Saddlepoint approximations for the probability of large deviations.* Preprint.
- Kolassa, J. E. (2003). Multivariate saddlepoint tail probability approximations. *Ann. Statist.* 31, 274-286.
- Kolassa, J. E. and Li, J. (2010). Multivariate saddlepoint approximations in tail probability and conditional inference. *Bernoulli* 16, 1191-1207.
- Wang, S. (1990). Saddlepoint approximations for bivariate distributions. *J. Appl. Probab.* 27, 586-597.
