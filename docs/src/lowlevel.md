# The derivative interface

The engine behind [`tailprob`](@ref) and [`density`](@ref) can be used directly with derivatives
of the cumulant generating function supplied by the user, for models whose derivatives are
available in closed form or by a recursion (large quadratic forms, for instance, where the trace
recursion of [`qf_Kmulti`](@ref) is faster than differentiating a log-determinant). These are the
functions of the replication material of the paper, with the same names and conventions.

The cumulant generating function is a function `Kfun(nu, t)` returning the derivative of
multi-index order `nu` (a tuple of `d` integers; `nu = (0, …, 0)` gives ``K(t)``) at the vector
`t`, and `NaN` outside the domain. A `CGF` can be turned into one with
`SaddlepointExpansions._kfun(K, maxdeg)` (internal).

| function | computes |
|---|---|
| [`spa_saddle`](@ref)`(Kfun, x)` | the saddlepoint, any dimension |
| [`spa_uni`](@ref)`(Kd, shat, xbar, order)` | ``d = 1``: density and distribution function from ``K^{(k)}(\hat s)``, ``k \le 2M+1`` |
| [`spa_ratio_pdf`](@ref)`(KWd, gd, shat, order)` | the Rice expansion of a ratio density |
| [`spa_bi`](@ref)`(Kfun, x, order)` | ``d = 2``: density and ``P(X \ge x)`` for ``\tau \ge 0`` |
| [`spa_bi_at`](@ref)`(Kd, tau, x, order)` | the same from the derivatives at a given saddlepoint |
| [`spa_orthant2`](@ref)`(Kfun, x, order)` | ``d = 2``: ``P(X \ge x)`` for any saddlepoint (inclusion-exclusion) |
| [`spa_mv`](@ref)`(Kfun, x, order)` | any ``d``: density and ``P(X \ge x)`` for ``\tau \ge 0`` |
| [`spa_tri`](@ref)`(Kfun, x, order)` | ``d = 3`` version of `spa_mv` |
| [`spa_mv_at`](@ref)`(Kd, tau, x, order)` | the same from the derivatives at a given saddlepoint |
| [`spa_orthant`](@ref)`(Kfun, x, order)` | any ``d``: ``P(X \ge x)`` for any saddlepoint |
| [`orthant2`](@ref), [`orthant3`](@ref) | normal orthant probabilities |

Whitened evaluation: `spa_bi` and `spa_mv` accept a function `Kwhite(nu, u, T)` returning the
derivatives of ``K_Y(u) = K(Tu)``; without it, the whitened Taylor coefficients are obtained by
substitution, and `coef_err_pdf`, `coef_err_cdf` in the result estimate the digits lost (the
`CGF` interface always supplies `Kwhite`).

## Closed-form derivative functions

| function | model |
|---|---|
| [`lin_exp_Kfun`](@ref)`(nu, t, A, n, lin)` | mean of ``n`` copies of ``A'U + \ell`` |
| [`ncx2_Kder`](@ref)`(u, m, lam, N)` | ``K^{(k)}(u)``, ``k \le N``, of ``\chi^2_m(\lambda)`` |
| [`bqfcc_Kfun`](@ref)`(nu, t, m, lam)`, [`mvcc_Kfun`](@ref) | ``Q_j = X_j + X_0`` with noncentral chi-squared parts |
| [`qf_Kmulti`](@ref)`(theta, Mats, L)` | all derivatives ``|\alpha| \le L`` of quadratic forms, by the trace recursion |
| [`wishdiag_kfun`](@ref)`(V, n)` | the diagonal of a Wishart matrix (memoized) |

## Ready-made univariate distributions

[`ncx2_spa`](@ref), [`ncf_spa`](@ref), [`dncf_spa`](@ref) (noncentral chi-squared, singly and
doubly noncentral ``F``) return the expansions of the distribution function, upper tail and
density, and [`ncx2_exact`](@ref), [`gamma_exact`](@ref), [`dncf_exact_dp`](@ref),
[`bqfcc_exact`](@ref), [`mvcc_exact`](@ref) give exact values to compare with.

## First-order approximations of the literature

[`kl_bivar`](@ref) (Kolassa and Li 2010) and [`wang_bivar`](@ref) (Wang 1990) evaluate the two
bivariate first-order approximations the paper compares with, from a `Kfun`. The first-order
term of [`tailprob`](@ref) is the approximation of Kolassa (2003).
