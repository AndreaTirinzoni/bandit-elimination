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

δs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
d = 10
K = 50
m = 4

# for reproducibility
rng = MersenneTwister(123)

arms = Vector{Float64}[]
v = zeros(d)
v[1] = 1.0
push!(arms, v)

θ = ones(d)

while length(arms) < K
    v = zeros(d)
    v2 = zeros(d)
    v3 = zeros(d)
    v[1] = 1
    while v'θ > 0.8 || v'θ < 0 
        v_small = 2 .* rand(rng, 3) .- 1   # Uniform in [-1,1]
        v_small ./= norm(v_small)   # make sure unit norm
        v = zeros(d)
        v2 = zeros(d)
        v3 = zeros(d)
        v[2:4] = v_small
        v2[5:7] = v_small
        v3[8:10] = v_small
    end
    push!(arms, v)
    push!(arms, v2)
    push!(arms, v3)
end

μ = [arm'θ for arm in arms]
topm_arms = istar(Topm(arms, m), θ)
println("min abs value of μ: ", minimum(abs.(μ)))
println("min gap: ", minimum(maximum(μ) .- maximum(μ[setdiff(1:K, Set([argmax(μ)]))])))
println("min gap topm: ", minimum(minimum(μ[topm_arms]) .- maximum(μ[setdiff(1:K, topm_arms)])))

# β = LinearThreshold(d, 1, 1, 1)
# β = GK16()
β = HeuristicThreshold()

pep = BAI(arms);

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
    @save isempty(ARGS) ? "experiments/results/delta_lin_$(typeof(pep))_$(typeof(sampling_rules[1]))_K$(K)_d$(d)_delta$(δ).dat" : ARGS[1] θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed

end

#################################################
# Oracle
#################################################
elim_rules = [NoElim(), CompElim()]
stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping())]
sampling_rules = [FixedWeights(w_star), FixedWeights(w_star)]

for conf in δs
    global δ = conf
    run()
end

#################################################
# LazyTaS
#################################################
elim_rules = [NoElim(), CompElim()]
stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping())]
sampling_rules = [LazyTaS(NoElimSR), LazyTaS(NoElimSR)]

for conf in δs
    global δ = conf
    run()
end

#################################################
# FWS
#################################################
elim_rules = [NoElim(), CompElim()]
stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping())]
sampling_rules = [FWSampling(NoElimSR), FWSampling(NoElimSR)]

for conf in δs
    global δ = conf
    run()
end

#################################################
# LinGapE
#################################################
elim_rules = [NoElim(), CompElim()]
stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping())]
sampling_rules = [LinGapE(NoElimSR), LinGapE(NoElimSR)]

for conf in δs
    global δ = conf
    run()
end

#################################################
# LinGame
#################################################
elim_rules = [NoElim(), CompElim()]
stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping())]
sampling_rules = [LinGame(CTracking, NoElimSR, false), LinGame(CTracking, NoElimSR, false)]

for conf in δs
    global δ = conf
    run()
end