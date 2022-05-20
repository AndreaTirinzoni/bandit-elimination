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
d = 40
K = 40
m = 3  # for topm

# for reproducibility
rng = MersenneTwister(123)

arms = Vector{Float64}[]
for k = 1:K
    v = zeros(d)
    v[k] = 1.0
    push!(arms, v)
end

θ = zeros(d)
θ[1] = 1
θ[2] = 0.9
θ[3] = 0.8
θ[4] = 0.7
θ[5] = 0.6
θ[6:end] = rand(rng, K-5) .* 0.5   # uniform in [0,0.5]

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
    @save isempty(ARGS) ? "experiments/results/uns_$(typeof(pep))_$(typeof(sampling_rules[1]))_K$(K)_d$(d).dat" : ARGS[1] θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed

end

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

#################################################
# LUCB
#################################################
elim_rules = [NoElim(), CompElim(), CompElim()]
stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),  Force_Stopping(max_samples, Elim_Stopping())]
sampling_rules = [LUCB(NoElimSR), LUCB(NoElimSR), LUCB(ElimSR)]

run()

#################################################
# UGapE
#################################################
elim_rules = [NoElim(), CompElim()]
stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping())]
sampling_rules = [UGapE(), UGapE()]

run()

#################################################
# Racing
#################################################
elim_rules = [NoElim()]
stopping_rules = [NoStopping()]
sampling_rules = [Racing()]

run()