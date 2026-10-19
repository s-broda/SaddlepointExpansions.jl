# Library of cumulant generating functions

Constructors returning a [`CGF`](@ref), written generically so that they can be differentiated
by Taylor-mode differentiation and evaluated in any precision.

| constructor | model |
|---|---|
| [`normal_cgf`](@ref)`(mu, Sigma)` | ``N(\mu, \Sigma)`` |
| [`gamma_cgf`](@ref)`(a, b)` | gamma with shape ``a`` and scale ``b``; `gamma_cgf(1, b)` is exponential |
| [`ncx2_cgf`](@ref)`(m, lam)` | noncentral chi-squared ``\chi^2_m(\lambda)`` |
| [`exponential_sums_cgf`](@ref)`(A; n, shift)` | mean of ``n`` copies of ``A'U + \ell``, ``U`` unit exponentials |
| [`quadratic_forms_cgf`](@ref)`(Mats; Sigma, mu, scale)` | quadratic forms ``z'A_iz`` in ``z \sim N(\mu, \Sigma)`` |
| [`wishart_diag_cgf`](@ref)`(V, n)` | the diagonal of ``W \sim W_d(n, V)`` |
| [`common_component_cgf`](@ref)`(m, lam)` | ``Q_j = X_j + X_0`` with independent noncentral chi-squared ``X_i`` |

The examples of the paper are: `exponential_sums_cgf([1 0; 1 1; 0 1]; n=10, shift=[-2, -2])`
(Wang 1990, Table 1), `exponential_sums_cgf([1 0; 1 1; 0 1]; n)` (Kolassa and Li 2010, Tables 3 to
6), `wishart_diag_cgf(0.75I + 0.25, n)` (Table 8), `common_component_cgf([m1, m2, m0], [λ1, λ2, λ0])`
(Table 9) and its trivariate version.

## Transformations

| function | result |
|---|---|
| [`reflect`](@ref)`(K, signs)` | ``(s_1X_1, …, s_dX_d)``, ``s_j = \pm1``; `reflect(K)` is ``-X`` |
| [`marginal`](@ref)`(K, idx)` | the components `idx` |
| [`linear_combination`](@ref)`(K, B)` | ``B'X`` for a ``d \times k`` matrix ``B`` |
| [`sample_mean`](@ref)`(K, n)` | the mean of ``n`` independent copies, ``nK(t/n)`` |
| [`shift`](@ref)`(K, c)` | ``X + c`` |
| `K1 + K2` | the sum of independent vectors |

```@docs
Base.:+(::CGF, ::CGF)
```

## Writing your own

Any generic Julia function works. Two patterns from the library:

```julia
# a sum of independent parts, one per component, plus a common part
K = CGF(t -> sum(-m[j]/2*log(1 - 2t[j]) for j in 1:d) - m0/2*log(1 - 2sum(t)), d;
        domain = t -> all(t .< 1/2) && sum(t) < 1/2)

# a log-determinant: factorize in generic arithmetic (see quadratic_forms_cgf), never with
# LinearAlgebra.logdet, which is not defined for Taylor polynomials
```

If the cumulant generating function is defined by a recursion or a numerical procedure that
cannot be differentiated this way, use the [derivative interface](lowlevel.md)
and supply the derivatives yourself.
