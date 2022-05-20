# On Elimination Strategies for Bandit Fixed-Confidence Identification

Code for the paper "On Elimination Strategies for Bandit Fixed-Confidence Identification" by Andrea Tirinzoni and Remy Degenne.

The code is written in Julia. It is an extension of public code from existing papers: 
 - [LinGame](https://github.com/xuedong/LinBAI.jl) (Degenne et al., 2020) 
 - [FWS](https://github.com/rctzeng/NeurIPS2021-Fast-Pure-Exploration-via-Frank-Wolfe) (Wang et al., 2021)
 - [RAGE](https://github.com/fiezt/Transductive-Linear-Bandit-Code) (Fiez et al., 2019)

## Setting up Julia

Please make sure that you have installed a version of Julia later or equal to 1.6.4. The code requires the following packages.
```
JLD2
StatsPlots
LaTeXStrings
IterTools
Distributions
JuMP
Tulip
```
To install them, run "julia" in a terminal and type "import Pkg; Pkg.add("PACKAGE NAME")".

## Code structure

The code is organized in the following files:

- [peps.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/peps.jl): it implements the three pure exploration problems we consider (BAI, Top-m, and OSI)
- [stopping_rules.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/stopping_rules.jl): it implements the *LLR* and *elimination* stopping rules
- [elimination_rules.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/elimination_rules.jl): it implements *selective* and *full* elimination for the different pure exploration problems as described in Appendix B
- [sampling_rules.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/sampling_rules.jl): it implements the sampling rules for all algorithms
- [envelope.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/envelope.jl): utils for FWS (extended from the [code](https://github.com/rctzeng/NeurIPS2021-Fast-Pure-Exploration-via-Frank-Wolfe) of Wang et al.)
- [regret.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/regret.jl): no-regret learners for LinGame
- [runit.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/runit.jl): functions to run an experiment
- [thresholds.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/thresholds.jl): different thresholds for stopping and/or sampling rules
- [tracking.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/tracking.jl): tracking rules for LinGame, TaS, and FWS
- [experiment_helpers.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiment_helpers.jl): some functions to plot and visualize results
- [utils.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/utils.jl): other general utilities

## Reproducing our experiments

There is one script for each experiment in the "experiments" folder which simply needs to be executed. The script will generate some .dat files with the results in the "results" sub-folder. Then, you can run the corresponding plot script in the "visualization" folder to visualize the results as in our paper. The scripts are:

- [linear_bai.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/linear_bai.jl), [linear_topm.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/linear_topm.jl), and [linear_osi.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/linear_osi.jl) for the experiments of Table 1. They can be visualized with [print_table.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/print_table.jl);
- [unstructured_bai.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/unstructured_bai.jl), [unstructured_topm.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/unstructured_topm.jl), and [unstructured_osi.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/unstructured_osi.jl) for the experiments of Table 4. They can be visualized with [print_table.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/print_table.jl)
- [linear_bai_elim.jl]() and [linear_topm_elim.jl]() for the experiments of Figure 1 (left and middle) and Table 3. They can be visualized with [plot_elim_algs.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/plot_elim_algs.jl), [plot_elim_full_vs_emp.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/plot_elim_full_vs_emp.jl), and [print_table_elim.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/print_table_elim.jl)
- [linear_bai_delta.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/linear_bai_delta.jl) for the experiment of Figure 1 (right). It can be visualized with [plot_delta.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/plot_delta.jl)
 
We also include a general visualization script [viz_experiment.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/viz_experiment.jl) which takes as input a .dat file (produced by the run of some algorithm) and plots the corresponding sample complexities, computation times, and elimination times.

**Note**: reproducing the experiments can take a long time. Most files run at least 5 algorithms, each with 3 to 5 variants and each for 100 runs, which makes roughly 2000 runs per file. To get faster results, just edit the script file to use less repetitions and/or to run only some desired algorithm. If one only cares about reproducing the plots, our results can be found in the experiments/saved_results folder.
