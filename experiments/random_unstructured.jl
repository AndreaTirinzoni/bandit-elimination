# Test LinGame on random unstructured instances

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

δ = 0.1
d = 10
K = d  # unstructured
m = 2  # for topm

# for reproducibility
rng = MersenneTwister(123)

arms = Vector{Float64}[]
for k = 1:d
    v = zeros(d)
    v[k] = 1.0
    push!(arms, v)
end

θ = 2 * rand(rng, d) .- 1  # sample θ uniformly in [-1,1]^d
θ += 0.05 * sign.(θ) # make sure every element is away from zero (for OSI)
topm_arms = istar(Topm(arms, m), θ)
θ[topm_arms] .+= 0.05  # make sure all best m arms have a minimum gap wrt the others (for Topm)
println("min abs value of θ: ", minimum(abs.(θ)))
println("min gap: ", minimum(maximum(θ) .- maximum(θ[setdiff(1:d, Set([argmax(θ)]))])))
println("min gap topm: ", minimum(minimum(θ[topm_arms]) .- maximum(θ[setdiff(1:d, topm_arms)])))

β = LinearThreshold(d, 1, 1, 1)

pep = BAI(arms);
pep = Topm(arms, m);
#pep = OSI(arms);

max_samples = 1e4
# elim_rules = [NoElim(), CompElim(), CompElim(), StatElim(), StatElim()]
# stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()),  Force_Stopping(max_samples, Elim_Stopping()), Force_Stopping(max_samples, Elim_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),Force_Stopping(max_samples, Elim_Stopping())]
# sampling_rules = [LinGame(CTracking, NoElimSR), LinGame(CTracking, NoElimSR), LinGame(CTracking, ElimSR), LinGame(CTracking, NoElimSR), LinGame(CTracking, ElimSR)]

elim_rules = [NoElim(), CompElim(), CompElim(), NoElim(), CompElim(), NoElim(), NoElim()]
stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()), Force_Stopping(max_samples, Elim_Stopping()), Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()), Force_Stopping(max_samples, LLR_Stopping()), NoStopping()]
sampling_rules = [LUCB(NoElimSR), LUCB(NoElimSR), LUCB(ElimSR), UGapE(), UGapE(), LinGapE(NoElimSR), Racing()]

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
@save isempty(ARGS) ? "experiments/results/rand_uns_$(typeof(pep))_K$(K).dat" : ARGS[1] θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed