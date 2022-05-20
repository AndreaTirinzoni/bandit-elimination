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

name = "lin_BAI_K50_d10_full_vs_emp"
file = "experiments/saved_results/elim/elim_lin_BAI_LinGame_K50_d10"
labels = ["Stopping (sel)", "Stopping + sampling (sel)", "Stopping (full)", "Stopping + sampling (full)"]

@load "$(file).dat" θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed

merged_data = Array{Tuple{Int64, Vector{Int64}, UInt64, Vector{Float64}}}(undef, 4, repeats)
merged_stopping = stopping_rules[2:5]
merged_sampling = sampling_rules[2:5]
merged_elim = elim_rules[2:5]
merged_data[1:4, :] = data[2:5, :]

plot(plot_elim_times(pep, θ, δ, β, merged_stopping, merged_sampling, merged_elim, merged_data, 50, labels, 7000));
plot!(guidefontsize=16, tickfontsize=12)
default(legendfontsize=16)
# plot!(xticks=[0,1e4,3e4,5e4])
xlabel!(L"\texttt{Time\ step}")
ylabel!(L"\texttt{Number\ of\ active\ arms}")
isdir("experiments/results") || mkdir("experiments/results")
savefig("experiments/results/$(name).pdf");