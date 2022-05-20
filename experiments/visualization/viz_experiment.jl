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

name = "experiments/results/rand_lin_BAI_K50_d20";

@load "$name.dat" θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed

dump_stats(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, repeats);

plot(boxes(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, "samples"));
ylabel!(L"\texttt{Sample\ complexity}")
isdir("experiments/results") || mkdir("experiments/results")
savefig("$(name)_samples.pdf");

plot(boxes(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, "time-iter"));
ylabel!(L"\texttt{Average\ iteration\ time\ (s)}")
savefig("$(name)_time.pdf");

plot(plot_elim_times(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, 50));
xlabel!(L"\texttt{Time\ step}")
ylabel!(L"\texttt{Number\ of\ active\ arms}")
savefig("$(name)_elim.pdf");
