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


struct MutableLinearIntersectionSystem{LIS <: LinearIntersectionSystem} <: HC.AbstractSystem
	F::LIS
end


function HC.cache(F::MutableLinearIntersectionSystem, x, args...)
    HC.cache(F.F, x)
end


function set_Ab!(F::LinearIntersectionSystem, p)
	# p == [vec(F.A);F.b]
	k = 1
	for j in 1:size(F.A,2), i in 1:size(F.A,1)
		F.A[i,j] = p[k]
		k += 1
	end
	for i in 1:size(F.b, 1)
		F.b[i] = p[k]
		k += 1
	end
	F
end




function HC.evaluate(F::MutableLinearIntersectionSystem, x, p, cache::LinearIntersectionSystemCache)
	set_Ab!(F.F, p)
	HC.evaluate(F.F, x, cache)
end

function HC.evaluate!(u, F::MutableLinearIntersectionSystem, x, p, cache::LinearIntersectionSystemCache)
	set_Ab!(F.F, p)
	HC.evaluate!(u, F.F, x, cache)
end


function HC.evaluate_and_jacobian(F::MutableLinearIntersectionSystem, x, p, cache::LinearIntersectionSystemCache)
	set_Ab!(F.F, p)
	HC.evaluate_and_jacobian(F.F, x, cache)
end

function HC.evaluate_and_jacobian!(u, U, F::MutableLinearIntersectionSystem, x, p, cache::LinearIntersectionSystemCache)
	set_Ab!(F.F, p)
	HC.evaluate_and_jacobian!(u, U, F.F, x, cache)
end

function HC.jacobian(F::MutableLinearIntersectionSystem, x, p, cache::LinearIntersectionSystemCache)
	set_Ab!(F.F, p)
    HC.jacobian(F.F, x, cache)
end

function HC.jacobian!(U, F::MutableLinearIntersectionSystem, x, p, cache::LinearIntersectionSystemCache)
	set_Ab!(F.F, p)
    HC.jacobian!(U, F.F, x, cache)
end


Base.size(F::MutableLinearIntersectionSystem) = size(F.F)

_return_type(F::MutableLinearIntersectionSystem, c::LinearIntersectionSystemCache) = _return_type(F.F, c)


HC.degrees(F::MutableLinearIntersectionSystem) = HC.degrees(F.F)

function HC.differentiate_parameters!(U, F::MutableLinearIntersectionSystem, x, p, c::LinearIntersectionSystemCache)
	U = zeros(_return_type(F, c), size(F.F,1), length(p))
	set_Ab!(F.F, p)

	jac = HC.jacobian!(U, F.F.F, F.F.A*x+F.F.b, c.cache)

	for i = 1:size(F.F.A, 1)
		for j = 1:size(F.F.A, 2)
			for k = 1:size(F.F, 1)
				U[k ,j*size(F.F,2)+i] += jac[k,i] * F.F.A[i,j]
			end
		end
	end

	for k = 1:size(F.F, 1)
		for i = 1:size(F.F, 2)
			U[k, length(p)-size(F.F,2)+i] += jac[k,i]
		end
	end

    U
end

function HC.differentiate_parameters(F::MutableLinearIntersectionSystem, x, p, c::LinearIntersectionSystemCache)
	U = similar(c.U, size(F.F, 1), length(p))
    HC.differentiate_parameters!(U , F, x, p, c)
end
