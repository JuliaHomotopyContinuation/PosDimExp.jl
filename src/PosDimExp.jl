module PosDimExp
import HomotopyContinuation
const HC = HomotopyContinuation
export witness_sup, witness, trace_list, WSet, trace, cache, LinearIntersectionSystem, MutableLinearIntersectionSystem

include("linear_sys.jl")
include("wset.jl")
include("trace.jl")
include("disj_set.jl")
include("irr_comp.jl")
include("alpha_ver.jl")
end
