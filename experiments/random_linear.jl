# Test LinGame on random linear instances

using JLD2;
using Distributed;
using Printf;
using IterTools;
using Distributions

include("../thresholds.jl")
include("../peps.jl")
include("../elimination_rules.jl")
include("../stopping_rules.jl")
include("../sampling_rules.jl")
include("../runit.jl")
include("../experiment_helpers.jl")
include("../utils.jl")
include("../envelope.jl")

δ = 0.1
d = 20
K = 50
m = 2  # for topm

# for reproducibility
rng = MersenneTwister(123)

arms = Vector{Float64}[]
for k = 1:K
    v = 2 * rand(rng, d) .- 1 # sample features in [-1,1]^d
    v[1] += 0.1 * sign(sum(v)) # make sure the final mean is away from zero (for OSI)
    push!(arms, v)
end

θ = ones(d)  # just take all ones since features are already random
μ = [arm'θ for arm in arms]
topm_arms = istar(Topm(arms, m), θ)
println("min abs value of μ: ", minimum(abs.(μ)))
println("min gap: ", minimum(maximum(μ) .- maximum(μ[setdiff(1:K, Set([argmax(μ)]))])))
println("min gap topm: ", minimum(minimum(μ[topm_arms]) .- maximum(μ[setdiff(1:K, topm_arms)])))

β = LinearThreshold(d, 1, 1, 1)
β = GK16()

pep = BAI(arms);
#pep = Topm(arms, m);
#pep = OSI(arms);

if typeof(pep) != Topm
    w_star = optimal_allocation(pep, θ, false, 10000)
    println("Optimal allocation: ", round.(w_star, digits=3))
end

max_samples = 1e9
#elim_rules = [NoElim(), CompElim(), CompElim(), StatElim(), StatElim()]
#stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()),  Force_Stopping(max_samples, Elim_Stopping()), Force_Stopping(max_samples, Elim_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),Force_Stopping(max_samples, Elim_Stopping())]

# LinGame
#sampling_rules = [LinGame(CTracking, NoElimSR), LinGame(CTracking, NoElimSR), LinGame(CTracking, ElimSR), LinGame(CTracking, NoElimSR), LinGame(CTracking, ElimSR)]

# LinGapE
#sampling_rules = [LinGapE(NoElimSR), LinGapE(NoElimSR), LinGapE(ElimSR), LinGapE(NoElimSR), LinGapE(ElimSR)]

# LinGame vs LinGapE
# elim_rules = [NoElim(), CompElim(), CompElim(), NoElim(), CompElim(), CompElim()]
# stopping_rules = [LLR_Stopping(), Elim_Stopping(), Elim_Stopping(), LLR_Stopping(), Elim_Stopping(), Elim_Stopping()]
# sampling_rules = [LinGame(CTracking, NoElimSR), LinGame(CTracking, NoElimSR), LinGame(CTracking, ElimSR), LinGapE(NoElimSR), LinGapE(NoElimSR), LinGapE(ElimSR)]

# FW-Sampling
# elim_rules = [NoElim(), CompElim(), CompElim()]
# stopping_rules = [LLR_Stopping(), Elim_Stopping(), Elim_Stopping()]
# sampling_rules = [FWSampling(NoElimSR), FWSampling(NoElimSR), FWSampling(ElimSR)]

# LinGame vs LinGapE vs FW-Sampling
# elim_rules = [NoElim(), CompElim(), NoElim(), CompElim(), NoElim(), CompElim()]
# stopping_rules = [LLR_Stopping(), Elim_Stopping(), LLR_Stopping(), Elim_Stopping(), LLR_Stopping(), Elim_Stopping()]
# sampling_rules = [LinGame(CTracking, NoElimSR), LinGame(CTracking, NoElimSR), LinGapE(NoElimSR), LinGapE(NoElimSR), FWSampling(), FWSampling()]

# Lazy-TaS
# elim_rules = [NoElim(), CompElim(), CompElim()]
# stopping_rules = [LLR_Stopping(), Elim_Stopping(), Elim_Stopping()]
# sampling_rules = [LazyTaS(NoElimSR), LazyTaS(NoElimSR), LazyTaS(ElimSR)]

# XYAdaptive
# elim_rules = [NoElim()]
# stopping_rules = [NoStopping()]
# sampling_rules = [XYAdaptive()]

# RAGE
# elim_rules = [NoElim()]
# stopping_rules = [NoStopping()]
# sampling_rules = [RAGE()]

# Oracle fixed weights
# elim_rules = [NoElim()]
# stopping_rules = [LLR_Stopping()]
# w = optimal_allocation(pep, θ, false, 10000)
# sampling_rules = [FixedWeights(w)]

# LinGIFA
# elim_rules = [NoElim(), NoElim()]
# stopping_rules = [LLR_Stopping(), LLR_Stopping()]
# sampling_rules = [LinGIFA(), LinGapE(NoElimSR)]

# Comparison of elimination times
# elim_rules = [CompElim(), CompElim(), CompElim(), CompElim()]
# stopping_rules = [Elim_Stopping(), Elim_Stopping(), Elim_Stopping(), Elim_Stopping()]
# sampling_rules = [LinGame(CTracking, NoElimSR), LinGame(CTracking, ElimSR), LinGapE(NoElimSR), LinGapE(ElimSR)]

# LinGame vs LinGapE vs FWS vs LazyTaS vs Oracle
elim_rules = [NoElim(), NoElim(), NoElim(), NoElim(), NoElim()]
stopping_rules = [LLR_Stopping(), LLR_Stopping(), LLR_Stopping(), LLR_Stopping(), LLR_Stopping()]
sampling_rules = [LinGame(CTracking, NoElimSR, false), LinGapE(NoElimSR), FWSampling(NoElimSR), LazyTaS(NoElimSR), FixedWeights(w_star)]

repeats = 10;
seed = 123;

# One fake run for each algorithm
# This is to have fair comparison of compute times later since Julia compiles stuff at the first calls
@time data = map(  # TODO: replace by pmap (it is easier to debug with map)
    ((sampling, stopping, elim),) -> runit(seed, sampling, stopping, elim, θ, pep, β, δ), 
    zip(sampling_rules, stopping_rules, elim_rules)
);

@time data = map(  # TODO: replace by pmap (it is easier to debug with map)
    (((sampling, stopping, elim), i),) -> runit(seed + i, sampling, stopping, elim, θ, pep, β, δ), 
    Iterators.product(zip(sampling_rules, stopping_rules, elim_rules), 1:repeats),
);

dump_stats(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, repeats);

# save
isdir("experiments/results") || mkdir("experiments/results")
@save isempty(ARGS) ? "experiments/results/rand_lin_$(typeof(pep))_K$(K)_d$(d).dat" : ARGS[1] θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed