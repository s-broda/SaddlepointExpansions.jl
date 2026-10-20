# Reference values of SaddlepointExpansions.jl for the test suites of the MATLAB and Python
# counterparts (SaddlepointExpansions-matlab/tests/test_spx.m, saddlepoint-expansions/tests).
# Writes reference_values.csv (columns case, quantity, order, value) next to this file; copy it to
# the other packages. The first part is the case list of R. Kan's julia_reference.jl, kept as it
# is so that the MATLAB tests keep working; the second part adds the cases of the Python tests.
#   julia --project=. test/reference_values.jl          (from the package folder)
using SaddlepointExpansions, LinearAlgebra, Printf

toep(r, n) = [r^abs(i - j) for i in 1:n, j in 1:n]
rows = String[]
add(case, q, v) = for (M, x) in enumerate(v); push!(rows, @sprintf("%s,%s,%d,%.17g", case, q, M, x)); end
add1(case, q, x) = push!(rows, @sprintf("%s,%s,1,%.17g", case, q, x))

# ===== the cases of the MATLAB tests (julia_reference.jl of the MATLAB package) =====
# 1. exponential sums, the README example
K = CGF(t -> -log(1 - t[1]) - log(1 - t[2]) - log(1 - t[1] - t[2]), 2;
        domain = t -> t[1] < 1 && t[2] < 1 && t[1] + t[2] < 1)
Kn = sample_mean(K, 5)
add("expsum2", "F", tailprob(Kn, [3.0, 3.0]; order=4).F)
add("expsum2", "f", density(Kn, [3.0, 3.0]; order=4).f)
add("expsum2", "Flow", tailprob(Kn, [1.5, 1.8]; order=4, lower=true).F)
add("expsum2", "Fmix", tailprob(Kn, [2.5, 1.5]; order=4, signs=[1, -1]).F)
# 2. gamma and noncentral chi-squared
add("gamma", "F", tailprob(gamma_cgf(5.0), 8.0; order=5).F)
add("gamma", "Flow", tailprob(gamma_cgf(5.0), 3.0; order=5, lower=true).F)
add("gamma", "f", density(gamma_cgf(5.0), 8.0; order=5).f)
add("ncx2", "F", tailprob(ncx2_cgf(10.0, 5.0), 25.0; order=5).F)
add("ncx2", "f", density(ncx2_cgf(10.0, 5.0), 25.0; order=5).f)
# 3. uniform mean of 4
Ku = sample_mean(uniform_cgf(), 4)
add("uniform", "F", tailprob(Ku, 0.4; order=4).F)
add("uniform", "f", density(Ku, 0.4; order=4).f)
# 4. common component, bivariate (Table 1 point) and trivariate
Kc = common_component_cgf([15.0, 20.0, 10.0], [5.0, 10.0, 8.0])
add("cc2", "F", tailprob(Kc, [48.09950493836207795336339, 65.23368793961408597955183]; order=5).F)
add("cc2", "f", density(Kc, [48.09950493836207795336339, 65.23368793961408597955183]; order=5).f)
Kc3 = common_component_cgf([4.0, 5.0, 6.0, 7.0])
add("cc3", "F", tailprob(Kc3, [14.0, 16.0, 17.0]; order=3).F)
add("cc3", "f", density(Kc3, [14.0, 16.0, 17.0]; order=3).f)
# 5. quadratic forms
n = 8
Kq = quadratic_forms_cgf([toep(0.5, n), Matrix(1.0I, n, n)]; Sigma=toep(0.3, n), mu=fill(0.5, n))
add("qf2", "F", tailprob(Kq, [14.0, 13.0]; order=4).F)
add("qf2", "f", density(Kq, [14.0, 13.0]; order=4).f)
D = derivatives(Kq, [0.05, -0.02], 5)
for nu in [(0, 0), (1, 0), (0, 1), (2, 1), (3, 2), (1, 4)]
    push!(rows, @sprintf("qf2,deriv_%d_%d,1,%.17g", nu[1], nu[2], D[nu]))
end
# 6. ratios
add("ratio_qf", "f", ratio_density(Kq, 1.2; order=4).f)
add("ratio_qf", "F", ratio_tailprob(Kq, 1.2; order=4).F)
add("ratio_qf", "Flow", ratio_tailprob(Kq, 0.9; order=4, lower=true).F)
# 7. Wishart diagonal and normal
Kw = wishart_diag_cgf([1.0 0.4; 0.4 2.0], 10)
add("wishart", "F", tailprob(Kw, [12.0, 25.0]; order=4).F)
Kz = normal_cgf([0.0, 1.0], [1.0 0.5; 0.5 2.0])
add("normal", "F", tailprob(Kz, [0.5, 1.5]; order=2).F)
# 8. exponential sums in three dimensions, saddlepoint with a negative component
A = [1.0 0 0; 0 1 0; 0 0 1; 1 1 0; 0 1 1]
Ke = exponential_sums_cgf(A; n=6)
add("expsum3", "F", tailprob(Ke, [2.2, 2.6, 1.9]; order=3).F)
add("expsum3", "f", density(Ke, [2.2, 2.6, 1.9]; order=3).f)

# ===== the cases of the Python tests =====
# Wang's example: X = U0+U1-2, Y = U0+U2-2, means of n = 10
Aw = [1.0 0; 1 1; 0 1]
Kwg = exponential_sums_cgf(Aw; n=10, shift=[-2.0, -2.0])
Dw = derivatives(Kwg, [0.3, -0.2], 7)
for (nu, v) in Dw
    push!(rows, @sprintf("wang,deriv_%d_%d,1,%.17g", nu[1], nu[2], v))
end
add("wang", "tau", saddlepoint(Kwg, [-0.5, -0.5])[1])
let r = tailprob(Kwg, [-0.5, -0.5]; order=5, lower=true)
    add("wang", "Flow", r.F); add("wang", "Flow_tau", r.tau); add("wang", "Flow_kappa", r.info.kappa)
end
let r = tailprob(Kwg, [-1.0, 2.0]; order=5, lower=true)
    add("wang", "Flow2", r.F); add("wang", "Flow2_tau", r.tau)
end
let r = tailprob(Kwg, [-0.8, 0.5]; order=3)
    add("wang", "Fie", r.F); add("wang", "Fie_tau", r.tau)
end
add("wang", "f", density(Kwg, [-0.5, -0.5]; order=4).f)
# Kolassa and Li's example, n = 5
K5 = exponential_sums_cgf(Aw; n=5)
for (name, x, order) in (("F33", [3.0, 3.0], 5), ("Fbdry", [2.5, 3.0], 5), ("Ffar", [6.0, 7.0], 4))
    r = tailprob(K5, x; order)
    add("kl", name, r.F); add("kl", name * "_tau", r.tau)
end
# pairs of quadratic forms with a common component
mm = [15.0, 20.0, 10.0]; ll = [5.0, 10.0, 8.0]
Kcc = common_component_cgf(mm, ll)
let r = tailprob(Kcc, [55.0, 70.0]; order=4)
    add("bqf", "F", r.F); add("bqf", "F_kappa", r.info.kappa)
end
add1("bqf", "F_exact", bqfcc_exact(55.0, 70.0, mm, ll).Fup)
add("bqf", "f", density(Kcc, [55.0, 70.0]; order=4).f)
add1("bqf", "f_exact", bqfcc_exact(55.0, 70.0, mm, ll).pdf)
add("bqf", "Flow", tailprob(Kcc, [40.0, 60.0]; order=3, lower=true).F)
add1("bqf", "Flow_exact", bqfcc_exact(40.0, 60.0, mm, ll).Flo)
# trivariate
m3 = [15.0, 20.0, 25.0, 10.0]; l3 = [5.0, 10.0, 0.0, 8.0]
K3 = common_component_cgf(m3, l3)
let r = tailprob(K3, [80.0, 95.0, 100.0]; order=4)
    add("tri", "F", r.F); add("tri", "F_err", r.info.err)
end
add1("tri", "F_exact", mvcc_exact([80.0, 95.0, 100.0], m3, l3).Fup)
let r = density(K3, [80.0, 95.0, 100.0]; order=4)
    add("tri", "f", r.f); add1("tri", "f_refined", r.info.refined ? 1.0 : 0.0)
end
add1("tri", "f_exact", mvcc_exact([80.0, 95.0, 100.0], m3, l3).pdf)
let r = tailprob(K3, [55.0, 70.0, 75.0]; order=3)
    add("tri", "Fie", r.F); add("tri", "Fie_tau", r.tau)
end
add1("tri", "Fie_exact", mvcc_exact([55.0, 70.0, 75.0], m3, l3).Fup)
# Wishart diagonal, 3 x 3
V = 0.75 .* Matrix(1.0I, 3, 3) .+ 0.25
Kwi = wishart_diag_cgf(V, 5)
Dwi = derivatives(Kwi, [0.05, -0.02, 0.03], 6)
for (nu, v) in Dwi
    push!(rows, @sprintf("wishart3,deriv_%d_%d_%d,1,%.17g", nu[1], nu[2], nu[3], v))
end
add("wishart3", "F", tailprob(Kwi, [6.5, 6.5, 7.5]; order=3).F)
# four dimensions
m4 = [15.0, 20.0, 25.0, 30.0, 10.0]; l4 = [5.0, 10.0, 0.0, 3.0, 8.0]
K4 = common_component_cgf(m4, l4); x4 = [70.0, 85.0, 95.0, 105.0]
let r = density(K4, x4; order=3)
    add("d4", "f", r.f); add1("d4", "f_refined", r.info.refined ? 1.0 : 0.0)
end
let r = tailprob(K4, x4; order=3)
    add("d4", "F", r.F); add("d4", "F_err", r.info.err)
end
add1("d4", "F_exact", mvcc_exact(x4, m4, l4).Fup)
# univariate
Kg = gamma_cgf(3.0)
add("gamma3", "F", tailprob(Kg, 6.0; order=4).F)
add("gamma3", "Flow", tailprob(Kg, 6.0; order=4, lower=true).F)
add("gamma3", "f", density(Kg, 6.0; order=4).f)
add1("gamma3", "tau", tailprob(Kg, 6.0; order=4).tau)
add("gamma3", "Ffar", tailprob(Kg, 20.0; order=5).F)
Kx = ncx2_cgf(5.0, 10.0)
add("ncx2b", "F", tailprob(Kx, 30.0; order=6).F)
add("ncx2b", "f", density(Kx, 30.0; order=6).f)
# the uniform distribution: the mean of 10 U(-1,1) variables
Ku10 = sample_mean(uniform_cgf(), 10)
add("uniform10", "F", tailprob(Ku10, 0.3; order=4).F)
add("uniform10", "f", density(Ku10, 0.3; order=4).f)
for (nu, v) in derivatives(uniform_cgf(), [0.0], 6)
    push!(rows, @sprintf("uniform10,deriv_%d,1,%.17g", nu[1], v))
end
let Kb = CGF(t -> log(sinhc(t[1])) + log(sinhc(t[1] + t[2])) + log(sinhc(t[2])), 2)
    r = tailprob(sample_mean(Kb, 8), [0.4, 0.3]; order=3)
    add("uniform10", "Fbi", r.F); add("uniform10", "Fbi_tau", r.tau)
end
# the noncentral F as a ratio
m_, n_, lam = 4, 12, 3; fq = 2.5; c = fq*m_/n_
KF = CGF(t -> ncx2_cgf(m_, lam)([t[1]]) + ncx2_cgf(n_)([t[2]]), 2; domain=t -> t[1] < 0.5 && t[2] < 0.5)
add("ncf", "f", ratio_density(KF, c; order=4).f)
add("ncf", "Flow", ratio_tailprob(KF, c; order=4, lower=true).F)
add("ncf", "F", ratio_tailprob(KF, c; order=4).F)
add1("ncf", "shat", ratio_density(KF, c; order=4).shat)
# mixed corner of a bivariate normal
S = [1.0 0.6; 0.6 2.0]
add("normal_mixed", "F", tailprob(normal_cgf([0.0, 0.0], S), [0.5, -0.3]; order=2, signs=[1, -1]).F)
add1("normal_mixed", "exact", orthant2([0.5, 0.3], [1.0 -0.6; -0.6 2.0]))
# two independent ratios of gamma variables, and a common denominator
a = [3.0, 4.0, 5.0, 6.0]
Kg4 = CGF(t -> -sum(a[i]*log(1 - t[i]) for i in 1:4), 4; domain=t -> all(t .< 1))
add("ratio2", "f", ratio_density(Kg4, [0.7, 0.9]; order=5).f)
add("ratio2", "Flow", ratio_tailprob(Kg4, [0.7, 0.9]; order=5, lower=true).F)
add("ratio2", "F", ratio_tailprob(Kg4, [0.7, 0.9]; order=5).F)
Kg3 = CGF(t -> -3log(1 - t[1]) - 4log(1 - t[2]) - 5log(1 - t[3]), 3; domain=t -> all(t .< 1))
add("ratio2", "f_common", ratio_density(Kg3, [0.5, 0.8]; order=4).f)
add("ratio2", "f_qf_general", ratio_density(Kq, 1.2; order=4, general=true).f)

open(joinpath(@__DIR__, "reference_values.csv"), "w") do io
    println(io, "case,quantity,order,value")
    foreach(r -> println(io, r), rows)
end
println("wrote ", length(rows), " rows")
