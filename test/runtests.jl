using Test, PosDimExp, HomotopyContinuation, LinearAlgebra, DynamicPolynomials

@testset "PosDimExp tests" begin
    @polyvar x[1:4]
    #F = SPSystem([x[1]*x[3]-x[2]^2; x[2]*x[4]-x[3]^2; x[1]*x[4]-x[2]*x[3]; x[4]-1])
    F = [  x[1]^3- x[2]*x[1] + 1; x[1]*x[3] + x[2]^3 + 0.5]
    w_sup = witness_sup(F, 1)

    @test length(w_sup.pts) == 9
    w_sup_broken = WSet(w_sup.F, w_sup.pts[1:2], w_sup.coords[1:2])
    @test abs(sum(trace_list(w_sup)))        ≈ 0 atol=1e-12
    @test abs(sum(trace_list(w_sup_broken))) > 1e-12
end

@testset "LinearIntersectionSystem tests" begin
    @polyvar x y z

    F     = SPSystem([x^4-1, y^3-1, z^2+x, x+y+z-1, x^2+z^2-3])
    w     = rand(ComplexF64, 2)
    A     = randn(ComplexF64, 3, 2)
    b     = randn(ComplexF64, 3)
    S     = LinearIntersectionSystem(F, A, b)
    system_cache = cache(S, w)

    @test system_cache isa AbstractSystemCache
    @test size(S) == (5, 2)
    @test evaluate(S, w, system_cache) ≈ evaluate(F, A * w + b)     atol=1e-12
    @test jacobian(S, w, system_cache) ≈ jacobian(F, A * w + b) * A atol=1e-12
    u, U = evaluate_and_jacobian(S, w, system_cache)
    @test u ≈ evaluate(S, w)
    @test U ≈ jacobian(S, w)

    S_mut = MutableLinearIntersectionSystem(S)
    parameters = [vec(A);b]
    system_cache = cache(S_mut, w)

    @test size(S_mut) == (5, 2)
    @test evaluate(S_mut, w, parameters, system_cache) ≈ evaluate(F, A * w + b)     atol=1e-12
    @test jacobian(S_mut, w, parameters, system_cache) ≈ jacobian(F, A * w + b) * A atol=1e-12
    u, U = evaluate_and_jacobian(S_mut, w, parameters, system_cache)
    @test u ≈ evaluate(S_mut, w, parameters, system_cache)
    @test U ≈ jacobian(S_mut, w, parameters, system_cache)
    @test rank(differentiate_parameters(S_mut, randn(ComplexF64, 2),[vec(S_mut.F.A); S_mut.F.b], system_cache)) == 3
end

@testset "α-test" begin
    @polyvar x y z
    L = [x^2*y + z; 2*x*y - 4 + z^2; x*y + y^2 - 2]
    F = SPSystem(L)
    pt = randn(ComplexF32, 3)
    pt2 = solutions(solve(L))[1]

    @test PosDimExp.alphatest(L, F, pt)  > 1e-12
    @test PosDimExp.alphatest(L, F, pt2) < 1e-12
end
