using JLD2;
using Printf;
using StatsPlots;
using LaTeXStrings;
using Statistics;

include("../../thresholds.jl")
include("../../peps.jl")
include("../../elimination_rules.jl")
include("../../stopping_rules.jl")
include("../../sampling_rules.jl")
include("../../experiment_helpers.jl")

normalize_samples = 1e3
normalize_times = 1e6

# BAI
files = ["experiments/saved_results/elim/elim_lin_BAI_LinGapE_K50_d10",
         "experiments/saved_results/elim/elim_lin_BAI_LinGame_K50_d10",
         "experiments/saved_results/elim/elim_lin_BAI_FWSampling_K50_d10",
         "experiments/saved_results/elim/elim_lin_BAI_LazyTaS_K50_d10",
         "experiments/saved_results/elim/elim_lin_BAI_FixedWeights_K50_d10", 
         "experiments/saved_results/elim/elim_lin_BAI_XYAdaptive_K50_d10",
         "experiments/saved_results/elim/elim_lin_BAI_RAGE_K50_d10"];

alg_names = ["LinGapE", "LinGame", "FWS", "Lazy TaS", "Oracle", "XY-Adaptive", "RAGE"]


# Top-m
files = ["experiments/saved_results/elim/elim_lin_Topm_LinGapE_K50_d10",
         "experiments/saved_results/elim/elim_lin_Topm_LinGame_K50_d10",
         "experiments/saved_results/elim/elim_lin_Topm_FWSampling_K50_d10",
         "experiments/saved_results/elim/elim_lin_Topm_LazyTaS_K50_d10",
         "experiments/saved_results/elim/elim_lin_Topm_FixedWeights_K50_d10",
         "experiments/saved_results/elim/elim_lin_Topm_LinGIFA_K50_d10"];

alg_names = ["m-LinGapE", "MisLid", "FWS", "Lazy TaS", "Oracle", "LinGIFA"]


for (f, alg) in zip(files, alg_names)
    @load "$f.dat" θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed
    row = "& " * alg * " & "

    samples = map(x -> sum(x[2])/normalize_samples, data); # (n_algos x n_repeats)
    samples_mean = round.(mean(samples, dims=2), digits=2)
    samples_std = round.(std(samples, dims=2), digits=1)

    times = map(x -> sum(x[3])/normalize_times, data) ./ map(x -> sum(x[2]), data); # (n_algos x n_repeats)
    times_mean = round.(mean(times, dims=2), digits=2)
    times_std = std(times, dims=2)

    if length(stopping_rules) == 1
        row *= "& & & & "
        row *= "\$$(samples_mean[1]) \\pm $(samples_std[1])\$ & \$$(times_mean[1])\$ & "
    elseif length(stopping_rules) == 3
        for i in 1:3
            row *= "\$$(samples_mean[i]) \\pm $(samples_std[i])\$ & \$$(times_mean[i])\$ & "
        end
    elseif length(stopping_rules) == 5
        for i in [1,2,4]
            row *= "\$$(samples_mean[i]) \\pm $(samples_std[i])\$ & \$$(times_mean[i])\$ & "
        end
        row = row[1:end-2]
        row *= "\\\\"
        println(row)
        row = "& " * alg * " + elim & & & "
        for i in [3,5]
            row *= "\$$(samples_mean[i]) \\pm $(samples_std[i])\$ & \$$(times_mean[i])\$ & "
        end
    else
        @assert false
    end

    row = row[1:end-2]
    row *= "\\\\"

    println(row)
end
