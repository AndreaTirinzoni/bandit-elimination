using JLD2;
using Printf;
using StatsPlots;
using LaTeXStrings;

include("../../thresholds.jl")
include("../../peps.jl")
include("../../elimination_rules.jl")
include("../../stopping_rules.jl")
include("../../sampling_rules.jl")
include("../../experiment_helpers.jl")

name = "lin_BAI_K50_d20"
files = ["experiments/saved_results/linear/lin_BAI_LinGapE_K50_d20",
         "experiments/saved_results/linear/lin_BAI_LinGame_K50_d20",
         "experiments/saved_results/linear/lin_BAI_FWSampling_K50_d20",
         #"experiments/saved_results/linear/lin_BAI_LazyTaS_K50_d20",
         "experiments/saved_results/linear/lin_BAI_FixedWeights_K50_d20"];
labels = ["LinGapE", "LinGapE + elim", "LinGame", "LinGame + elim", "FWS", "FWS + elim", "Oracle"]

@load "$(files[1]).dat" θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed

merged_data = Array{Tuple{Int64, Vector{Int64}, UInt64, Vector{Float64}}}(undef, 2 * (length(files)-1) + 1, repeats)
merged_stopping = []
merged_sampling = []
merged_elim = []

idx = 1
for f in files
    @load "$f.dat" θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed
    # dump_stats(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, repeats);

    push!(merged_stopping, stopping_rules[2])
    push!(merged_sampling, sampling_rules[2])
    push!(merged_elim, elim_rules[2])

    if labels[idx] == "Oracle"
        merged_data[idx, :] = data[2, :]
        global idx += 1
    else
        merged_data[idx:idx+1, :] = data[2:3, :]
        global idx += 2

        push!(merged_stopping, stopping_rules[3])
        push!(merged_sampling, sampling_rules[3])
        push!(merged_elim, elim_rules[3])
    end
end

plot(plot_elim_times(pep, θ, δ, β, merged_stopping, merged_sampling, merged_elim, merged_data, 50, labels, 6*1e4));
plot!(guidefontsize=16, tickfontsize=12)
default(legendfontsize=16)
plot!(xticks=[0,1e4,3e4,5e4])
xlabel!(L"\texttt{Time\ step}")
ylabel!(L"\texttt{Number\ of\ active\ arms}")
isdir("experiments/results") || mkdir("experiments/results")
savefig("experiments/results/$(name)_elim.pdf");