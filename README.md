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

## Reproducing our experiments

Reproducing our experiments is very simple. There is one script for each of them in the "experiments" folder which simply needs to be executed. The script will generate some .dat files with the results in the "results" sub-folder. Then, you can run the corresponding plot script in the "visualization" folder to visualize the results as in our paper. The scripts are:

- [linear_bai.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/linear_bai.jl), [linear_topm.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/linear_topm.jl), and [linear_osi.jl]() for the experiments of Table 1. They can be visualized with print_table.jl;
- unstructured_bai.jl, unstructured_topm.jl, and unstructured_osi.jl for the experiments of Table 4. They can be visualized with print_table.jl
- linear_bai_elim.jl and linear_topm_elim.jl for the experiments of Figure 1 (left and middle) and Table 3. They can be visualized with plot_elim_algs.jl, plot_elim_full_vs_emp.jl, and print_table_elim.jl
- linear_bai_delta.jl for the experiment of Figure 1 (right). It can be visualized with plot_delta.jl
 
There are also scripts random_linear.jl, random_unstructured.jl, and hard_linear.jl to run general tests. They can be visualized with the general script viz_experiment.jl, which simply takes as input a .dat file and plots the corresponding sample complexities, computation times, and elimination times.

NOTE: reproducing the experiments can take a long time. Most files run at least 5 algorithms, each with 3 to 5 variants and each for 100 runs, which makes roughly 2000 runs per file. To get faster results, just edit the script file to use less repetitions and to run only some desired algorithm. Our results are already in the experiments/saved_results folder.
