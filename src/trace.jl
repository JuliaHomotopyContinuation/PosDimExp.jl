using LinearAlgebra
"""
    `trace_list(witn)`

`witn` is struct of type `WSet`.
Returns a complex number.

`witn` represents a potential witness set that is tested for completeness.
Returns a list of traces.
"""
function trace_list(witn::WSet)
    rndn_dir, u = randn(ComplexF64, HC.size(witn.F.A, 1)) , rand(ComplexF64, 1, HC.size(witn.F.A, 1))
    s_1, s_2 = rand(), -rand()
    return trace(witn.F, witn.pts, witn.coords, s_1, s_2, rndn_dir, u)
end


function trace(F::LinearIntersectionSystem, pts, coords, s_1, s_2, rndn_dir, u)
    #Define a system G of affine linear maps.
    #This system defines the affine linear space of the potential witness set, translated into the random direction rnd_dir:
    p = [vec(F.A );F.b]
    p_1 = [vec(F.A );F.b + rndn_dir * s_1]
    p_2 = [vec(F.A );F.b + rndn_dir * s_2]
    F_mu = MutableLinearIntersectionSystem(LinearIntersectionSystem(F.F, copy(F.A), copy(F.b)))

    Y_1 = solutions(solve(F_mu, coords; start_parameters = p, target_parameters = p_1 ))
    Y_2 = solutions(solve(F_mu, coords; start_parameters = p, target_parameters = p_2 ))

    #From the obtained coordinates we reconstruct the actual witness sets:
    Y_1 = map( u -> F.A * u + F.b + rndn_dir.*s_1, Y_1 )
    Y_2 = map( u -> F.A * u + F.b + rndn_dir.*s_2, Y_2 )

    b, b_1, b_2 = [u ⋅ y for y in pts], [u ⋅ y for y in Y_1], [u ⋅ y for y in Y_2]
    return (b_1 - b)/s_1 - (b_2 - b)/s_2
end
