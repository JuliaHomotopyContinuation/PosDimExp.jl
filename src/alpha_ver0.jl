using LinearAlgebra, DynamicPolynomials

function β(F, x, jac_inv = nothing)
    if jac_inv == nothing
        jac_inv = inv(HC.jacobian(F,x))
    end
    euclidean_norm(jac_inv * evaluate(F, x))
end

function μ(L, F, x, M = nothing)
    degrees = HC.maxdegrees(L)
    nrm = norm_1(x)

    if M == nothing
        M = inv(HC.jacobian(F,x))
    end

    for i = 1:size(M,1)
        for j = size(M,2)
            M[i,j] *= 1/(sqrt(degrees[i]) * nrm^(degrees[i]-1))
        end
    end
    nrm_M_inv = norm(M)
    #ineffizient, berechnet alle Singulärwerte, nicht nur einen minimalen.

    nrm_F = 0
    for i = 1:length(L)
        nrm_F += pol_norm(L[i])
    end
    nrm_F = sqrt(nrm_F)

    max(1, nrm_F * nrm_M_inv)
end

function norm_1(x)
    nrm = 1
    for i = 1:length(x)
        nrm += abs2(x[i])
    end
    sqrt(nrm)
end

function alphatest(L, F::AbstractSystem, x)
    jac_inv = inv(HC.jacobian(F,x))
    α_bound = β(F, x, jac_inv) * μ(L, F, x, jac_inv) * maximum(HC.maxdegrees(L))^(3/2) / (2 * norm_1(x))
    α_bound
    # < 0.157671 ?!
end


function pol_norm(f)
    d = DynamicPolynomials.maxdegree(f)
    coeff = coefficients(f)
    nrm = 0

    for i = 1:length(f)
        mult_degr = exponents(f[i])
        nrm += abs2(coeff[i]) * (( prod(factorial.(mult_degr)) * factorial(d - sum(mult_degr)) )/ factorial(d))
    end
    sqrt(nrm)
end
