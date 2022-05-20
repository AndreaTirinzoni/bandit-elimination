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

normalize_samples = 1
δs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

name = "lin_BAI_K50_d10"

files_lge = ["experiments/saved_results/delta/delta_lin_BAI_LinGapE_K50_d10_delta$(δ)" for δ in δs];
files_lg = ["experiments/saved_results/delta/delta_lin_BAI_LinGame_K50_d10_delta$(δ)" for δ in δs];
files_fws = ["experiments/saved_results/delta/delta_lin_BAI_FWSampling_K50_d10_delta$(δ)" for δ in δs];
files_tas = ["experiments/saved_results/delta/delta_lin_BAI_LazyTaS_K50_d10_delta$(δ)" for δ in δs];
files_oracle = ["experiments/saved_results/delta/delta_lin_BAI_FixedWeights_K50_d10_delta$(δ)" for δ in δs];

file_lists = [files_lge, files_lg, files_fws, files_tas, files_oracle]
labels = ["LinGapE", "LinGame", "FWS", "Lazy TaS", "Oracle"]

@load "$(files_oracle[1]).dat" θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed

plt = plot(legend=:topright)

for (j,files) in enumerate(file_lists)

    deltas = zeros(length(files))
    means_llr = zeros(length(files))
    std_llr = zeros(length(files))
    means_elim = zeros(length(files))
    std_elim = zeros(length(files))

    means_ratios = zeros(length(files))
    std_ratios = zeros(length(files))

    for (i,f) in enumerate(files)
        @load "$f.dat" θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed
        # dump_stats(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, repeats);

        deltas[i] = δ

        samples = map(x -> sum(x[2])/normalize_samples, data); # (n_algos x n_repeats
        samples_mean = round.(mean(samples, dims=2), digits=2)
        samples_std = round.(std(samples, dims=2), digits=1)

        means_llr[i] = samples_mean[1]
        std_llr[i] = samples_std[1]
        means_elim[i] = samples_mean[2]
        std_elim[i] = samples_std[2]

        ratios = samples[1, :] ./ samples[2, :]
        means_ratios[i] = mean(ratios)
        std_ratios[i] = std(ratios)
    end

    deltas = log.(1 ./ deltas)
    std_ratios ./= sqrt(repeats)

    println(labels[j])
    println("LLR ", means_llr, " ", std_llr)
    println("Elim ", means_elim, " ", std_elim)
    println("Ratios ", means_ratios, " ", std_ratios)
    println()

    plot!(deltas, means_ratios, yerr=std_ratios, msc=j, label=labels[j], linewidth=2)

end

plot!(guidefontsize=16, tickfontsize=12)
default(legendfontsize=16)
plot!(yticks=[1.00, 1.03, 1.06, 1.09, 1.12])
xlabel!(L"$\log(1/\delta)$")
ylabel!(L"\texttt{Ratio\ of\ stopping\ times}")
isdir("experiments/results") || mkdir("experiments/results")
savefig("experiments/results/$(name)_delta.pdf");