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

δ = 0.01
d = 20
K = 50
m = 5  # for topm

# for reproducibility
rng = MersenneTwister(123)

arms = Vector{Float64}[]
mid = Int(ceil(d/2))
for k = 1:mid
    v = zeros(d)
    v[k] = 1.0
    push!(arms, v)
end

θ = zeros(d)
θ[1:mid] = rand(rng, mid) .* 0.1 .+ 0.1  # uniform in [0.1,0.2]
θ[1:mid] .*= sign.(2 .* rand(rng, mid) .- 1)  # randomize sign
θ[mid+1:end] = rand(rng, d-mid) .- 1   # uniform in [-0.5,0.5]

while length(arms) < K
    v = zeros(d)
    while abs(v'θ) < 0.5
        v = 2 .* rand(rng, d) .- 1   # Uniform in [-1,1]
        v ./= norm(v)   # make sure unit norm
    end
    push!(arms, v)
end

μ = [arm'θ for arm in arms]
topm_arms = istar(Topm(arms, m), θ)
println("min abs value of μ: ", minimum(abs.(μ)))
println("min gap: ", minimum(maximum(μ) .- maximum(μ[setdiff(1:K, Set([argmax(μ)]))])))
println("min gap topm: ", minimum(minimum(μ[topm_arms]) .- maximum(μ[setdiff(1:K, topm_arms)])))

# β = LinearThreshold(d, 1, 1, 1)
# β = GK16()
β = HeuristicThreshold()

pep = OSI(arms);

w_star = optimal_allocation(pep, θ, false, 10000)
println("Optimal allocation: ", round.(w_star, digits=3))

max_samples = 1e6

repeats = 100;
seed = 123;

function run()

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
    @save isempty(ARGS) ? "experiments/results/lin_$(typeof(pep))_$(typeof(sampling_rules[1]))_K$(K)_d$(d).dat" : ARGS[1] θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed

end

#################################################
# LinGapE
#################################################
elim_rules = [NoElim(), CompElim(), CompElim()]
stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),  Force_Stopping(max_samples, Elim_Stopping())]
sampling_rules = [LinGapE(NoElimSR), LinGapE(NoElimSR), LinGapE(ElimSR)]

run()

#################################################
# LinGame
#################################################
elim_rules = [NoElim(), CompElim(), CompElim()]
stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),  Force_Stopping(max_samples, Elim_Stopping())]
sampling_rules = [LinGame(CTracking, NoElimSR, false), LinGame(CTracking, NoElimSR, false), LinGame(CTracking, ElimSR, false)]

run()

#################################################
# Oracle
#################################################
elim_rules = [NoElim(), CompElim()]
stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping())]
sampling_rules = [FixedWeights(w_star), FixedWeights(w_star)]

run()

#################################################
# LazyTaS
#################################################
elim_rules = [NoElim(), CompElim(), CompElim()]
stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),  Force_Stopping(max_samples, Elim_Stopping())]
sampling_rules = [LazyTaS(NoElimSR), LazyTaS(NoElimSR), LazyTaS(ElimSR)]

run()

#################################################
# FWS
#################################################
elim_rules = [NoElim(), CompElim(), CompElim()]
stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),  Force_Stopping(max_samples, Elim_Stopping())]
sampling_rules = [FWSampling(NoElimSR), FWSampling(NoElimSR), FWSampling(ElimSR)]

run()