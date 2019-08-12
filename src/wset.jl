
using HomotopyContinuation, LinearAlgebra
HC = HomotopyContinuation

"""
    gen_rank(F; pt)

Computes the generic rank of the Jacobian of an abstract system `F`.

    * F
    * pt

Returns an integer.
"""
function gen_rank(F; pt = nothing)
    #Choose a random point if none was provided:
    if pt == nothing
        pt = randn(ComplexF64, size(F, 2))
    end
    rank(HC.jacobian(F, pt))
end

"""
    `struct WSet`

Represents a witness set of the algebraic set described by the abstract system `F`.
`pts` is the intersection of this algebraic set with the generic affine linear space
that is spanned by the row vectors of the matrix `A`, translated by the vector `b`.
"""
struct WSet
    F::LinearIntersectionSystem
    pts
    coords
end

"""
    `witness_sup(F, i)`

Given an abstract system 'F', compute a superset of an i-dimensional witness set.
If `F` describes a puredimensional algebraic set, then returns a witness set.
    * F
    * i
Returns struct `WSet`.
"""
function witness_sup(F, i)
    F_stat = SPSystem(F)
    pts, coords = [], []
    N = size(F_stat, 2)
    A, b = randn(ComplexF64, N,N-i), randn(ComplexF64, N)

    #Check whether 'F' describes the whole space
    if i == N
        pt = randn(ComplexF64, N)
        if norm(HC.evaluate(F_stat, pt)) > 1e-12
            return WSet(LinearIntersectionSystem(F_stat, A, b), pts, coords)
        end
        push!(pts, pt)
        return WSet(LinearIntersectionSystem(F_stat, A, b), pts, coords)
    end

    #If i is too small, get the empty set:
    if i < N - gen_rank(F_stat)
        return WSet(LinearIntersectionSystem(F_stat, A, b), pts, coords)
    end
"""
    @polyvar u[1:N-i]
    comp = F ∘ (A * u + b)
    all_coords = HC.solutions(HC.solve(comp))
"""
    F_lin = LinearIntersectionSystem(F_stat, A, b)
    all_coords = HC.solutions(HC.solve(F_lin))
    #Sort out junk due to randomization.
    for coord in all_coords
        pt = A * coord + b
        if norm(HC.evaluate(F_stat, pt)) <  1e-12
            push!(pts , pt)
            push!(coords, coord)
        end
    end
    return WSet(F_lin, pts, coords)
end
