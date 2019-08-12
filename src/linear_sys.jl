export LinearIntersectionSystem, evaluate, jacobian, evaluate_and_jacobian, cache

struct LinearIntersectionSystem{System <:HC.AbstractSystem,T} <: HC.AbstractSystem
	F::System
	A::Matrix{T}
	b::Vector{T}
end


struct LinearIntersectionSystemCache{SC<:HC.AbstractSystemCache, T} <: HC.AbstractSystemCache
    u::Vector{T}
    U::Matrix{T}
    cache::SC
end

function HC.cache(F::LinearIntersectionSystem, x)
    c = HC.cache(F.F, x)
    u = evaluate(F.F, x, c)
    U = similar(u, size(F.F))

    LinearIntersectionSystemCache(u, U, c)
end



function HC.evaluate!(u, F::LinearIntersectionSystem, x, c::LinearIntersectionSystemCache)
    evaluate!(c.u, F.F, F.A * x + F.b, c.cache)
	for i = 1: length(u)
		u[i] = c.u[i]
	end
    u
end

function HC.jacobian!(U, F::LinearIntersectionSystem, x, c::LinearIntersectionSystemCache)
    HC.jacobian!(c.U, F.F, F.A * x + F.b, c.cache)
	mul!(U, c.U, F.A)
	U
end

function HC.evaluate_and_jacobian!(u, U, F::LinearIntersectionSystem, x, c::LinearIntersectionSystemCache)
    evaluate_and_jacobian!(c.u, c.U, F.F, F.A * x + F.b, c.cache)
	for i = 1: length(u)
		u[i] = c.u[i]
	end
	mul!(U, c.U, F.A)
    nothing
end

function HC.evaluate(F::LinearIntersectionSystem, x, c::LinearIntersectionSystemCache)
    u = Vector{_return_type(F,c)}(undef, size(F.F, 1))
    evaluate!(u, F, x, c)
end

function HC.jacobian(F::LinearIntersectionSystem, x, c::LinearIntersectionSystemCache)
    U = Matrix{_return_type(F,c)}(undef, size(F.F, 1), size(F.A)[2])
    jacobian!(U, F, x, c)
end

function HC.evaluate_and_jacobian(F::LinearIntersectionSystem, x, c::LinearIntersectionSystemCache)
    T = _return_type(F,c)
    u = Vector{T}(undef, size(F.F, 1))
    U = Matrix{T}(undef, size(F.F, 1), size(F.A)[2])
    evaluate_and_jacobian!(u, U, F, x, c)
    u, U
end

Base.size(F::LinearIntersectionSystem) = (size(F.F)[1], size(F.A)[2])

_return_type(F::LinearIntersectionSystem, c::LinearIntersectionSystemCache) = typeof(F.A[1,1] * c.u[1])

HC.degrees(F::LinearIntersectionSystem) = HC.degrees(F.F)
