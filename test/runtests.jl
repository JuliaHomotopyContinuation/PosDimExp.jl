using Test, PosDimExp, HomotopyContinuation

@testset "PosDimExp tests" begin
    @polyvar x[1:4]
    F = SPSystem([x[1]*x[3]-x[2]^2; x[2]*x[4]-x[3]^2; x[1]*x[4]-x[2]*x[3]; x[4]-1])
    #F = [  x[1]^3- x[2]*x[1] + 1; x[1]*x[3] + x[2] + 0.5; x[1] + x[2]^3 ]
    w_sup = witness_sup(F, 0)

    @test length(w_sup.pts) == 3
    w_sup_broken = WSet(w_sup.F, w_sup.pts[1:2], w_sup.coords[1:2])
    @test abs(sum(trace_list(w_sup)))        ≈ 0 atol=1e-12
    @test abs(sum(trace_list(w_sup_broken))) > 1e-12
end

@testset "LinearIntersectionSystem tests" begin
    @polyvar x y z

    F = SPSystem([x^4-1, y^3-1, z^2+x, x+y+z-1, x^2+z^2-3])
    w = rand(ComplexF64, 2)
    A = randn(ComplexF64, 3, 2)
    b = randn(ComplexF64, 3)
    S = LinearIntersectionSystem(F, A, b)
    system_cache = cache(S, w)
    @test system_cache isa AbstractSystemCache

    @test size(S) == (5, 2)
    @test evaluate(S, w, system_cache) ≈ evaluate(F, A * w + b)     atol=1e-12
    @test jacobian(S, w, system_cache) ≈ jacobian(F, A * w + b) * A atol=1e-12
    u, U = evaluate_and_jacobian(S, w, system_cache)
    @test u ≈ evaluate(S, w)
    @test U ≈ jacobian(S, w)
end
