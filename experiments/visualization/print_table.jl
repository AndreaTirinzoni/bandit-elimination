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

# # BAI lin
# files = ["experiments/saved_results/linear/lin_BAI_LinGapE_K50_d20",
#          "experiments/saved_results/linear/lin_BAI_LinGame_K50_d20",
#          "experiments/saved_results/linear/lin_BAI_FWSampling_K50_d20",
#          "experiments/saved_results/linear/lin_BAI_LazyTaS_K50_d20",
#          "experiments/saved_results/linear/lin_BAI_FixedWeights_K50_d20", 
#          "experiments/saved_results/linear/lin_BAI_XYAdaptive_K50_d20",
#          "experiments/saved_results/linear/lin_BAI_RAGE_K50_d20"];

# alg_names = ["LinGapE", "LinGame", "FWS", "Lazy TaS", "Oracle", "XY-Adaptive", "RAGE"]

# # BAI uns
# files = ["experiments/saved_results/unstructured/uns_BAI_LinGame_K40_d40",
#          "experiments/saved_results/unstructured/uns_BAI_FWSampling_K40_d40",
#          "experiments/saved_results/unstructured/uns_BAI_LazyTaS_K40_d40",
#          "experiments/saved_results/unstructured/uns_BAI_FixedWeights_K40_d40", 
#          "experiments/saved_results/unstructured/uns_BAI_LUCB_K40_d40",
#          "experiments/saved_results/unstructured/uns_BAI_UGapE_K40_d40",
#          "experiments/saved_results/unstructured/uns_BAI_Racing_K40_d40"];

# alg_names = ["k-Learner", "FWS", "Lazy TaS", "Oracle", "LUCB", "UGapE", "Racing"]

# # Top-m
# files = ["experiments/saved_results/linear/lin_Topm_LinGapE_K50_d20",
#          "experiments/saved_results/linear/lin_Topm_LinGame_K50_d20",
#          "experiments/saved_results/linear/lin_Topm_FWSampling_K50_d20",
#          "experiments/saved_results/linear/lin_Topm_LazyTaS_K50_d20",
#          "experiments/saved_results/linear/lin_Topm_FixedWeights_K50_d20", 
#          "experiments/saved_results/linear/lin_Topm_LinGIFA_K50_d20"];

# alg_names = ["m-LinGapE", "MisLid", "FWS", "Lazy TaS", "Oracle", "LinGIFA"]

# # Top-m uns
# files = ["experiments/saved_results/unstructured/uns_Topm_LinGame_K40_d40",
#          "experiments/saved_results/unstructured/uns_Topm_FWSampling_K40_d40",
#          "experiments/saved_results/unstructured/uns_Topm_LazyTaS_K40_d40",
#          "experiments/saved_results/unstructured/uns_Topm_FixedWeights_K40_d40", 
#          "experiments/saved_results/unstructured/uns_Topm_LUCB_K40_d40",
#          "experiments/saved_results/unstructured/uns_Topm_UGapE_K40_d40",
#          "experiments/saved_results/unstructured/uns_Topm_Racing_K40_d40"];

# alg_names = ["k-Learner", "FWS", "Lazy TaS", "Oracle", "LUCB", "UGapE", "Racing"]

# # OSI
# files = ["experiments/saved_results/linear/lin_OSI_LinGapE_K50_d20",
#          "experiments/saved_results/linear/lin_OSI_LinGame_K50_d20",
#          "experiments/saved_results/linear/lin_OSI_FWSampling_K50_d20",
#          "experiments/saved_results/linear/lin_OSI_LazyTaS_K50_d20",
#          "experiments/saved_results/linear/lin_OSI_FixedWeights_K50_d20"];

# alg_names = ["LinGapE", "LinGame", "FWS", "Lazy TaS", "Oracle"]

files = ["experiments/saved_results/unstructured/uns_OSI_LinGame_K40_d40",
         "experiments/saved_results/unstructured/uns_OSI_FWSampling_K40_d40",
         "experiments/saved_results/unstructured/uns_OSI_LazyTaS_K40_d40",
         "experiments/saved_results/unstructured/uns_OSI_FixedWeights_K40_d40", 
         "experiments/saved_results/unstructured/uns_OSI_LUCB_K40_d40"];

alg_names = ["k-Learner", "FWS", "Lazy TaS", "Oracle", "LUCB"]


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
    end

    for i in 1:length(stopping_rules)
        row *= "\$$(samples_mean[i]) \\pm $(samples_std[i])\$ & \$$(times_mean[i])\$ & "
    end

    if length(stopping_rules) == 3 || length(stopping_rules) == 1
        row = row[1:end-2]
        row *= "\\\\"
    elseif length(stopping_rules) == 2
        row *= "& \\\\"
    else
        @assert false
    end

    println(row)
end
