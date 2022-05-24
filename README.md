# Elimination Strategies for Bandit Identification

Code for the paper "On Elimination Strategies for Bandit Fixed-Confidence Identification" by Andrea Tirinzoni and Remy Degenne. A preprint is available [here](https://arxiv.org/abs/2205.10936).

The code is written in Julia. It is an extension of public code from existing papers: 
 - [LinGame](https://github.com/xuedong/LinBAI.jl) by Degenne et al., 2020 [1]
 - [FWS](https://github.com/rctzeng/NeurIPS2021-Fast-Pure-Exploration-via-Frank-Wolfe) by Wang et al., 2021 [2]
 - [RAGE](https://github.com/fiezt/Transductive-Linear-Bandit-Code) by Fiez et al., 2019 [3]

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

## Implemented algorithms

The table below summarizes all the algorithms available in this repository. For each of them, we specify whether it works in unstructured and/or linear bandit instances, what pure exploration problems it can handle, whether it can be combined with our elimination rules at stopping and/or sampling, and whether it is natively elimination based. The oracle algorithm in the last row simply plays the fixed optimal proportions from the lower bound.

| Name        | Unstructured | Linear | BAI | Top-m | OSI | Elim. stopping | Elim. sampling | Native elim. |
| ----------- | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| LinGame [1] | | X | X | | X | X | X | |
| FWS [2] | X | X | X | X | X | X | X | |
| RAGE [3] | | X | X | | | | | X |
| Lazy TaS [4] | X | X | X | X | X | X | X | |
| XY-Adaptive [5] | | X | X | | | | | X |
| m-LinGapE [6] | | X | | X | | X | X | |
| MisLid [7] | | X | | X | | X | X | |
| LinGIFA [6] | | X | | X | | X | | |
| k-Learner [8] | X | | X | X | X | X | X | |
| LUCB [9] | X | | X | X | X | X | X | |
| UGapE [10] | X | | X | X | | X | | |
| Racing [11] | X | | X | X | | | | X |
| Oracle | X | X | X | X | X | X | X | |


## Reproducing our experiments

There is one script for each experiment in the "experiments" folder which simply needs to be executed. The script will generate some .dat files with the results in the "results" sub-folder. Then, you can run the corresponding plot script in the "visualization" folder to visualize the results as in our paper. The scripts are:

- [linear_bai.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/linear_bai.jl), [linear_topm.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/linear_topm.jl), and [linear_osi.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/linear_osi.jl) for the experiments of Table 1. They can be visualized with [print_table.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/print_table.jl);
- [unstructured_bai.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/unstructured_bai.jl), [unstructured_topm.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/unstructured_topm.jl), and [unstructured_osi.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/unstructured_osi.jl) for the experiments of Table 4. They can be visualized with [print_table.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/print_table.jl)
- [linear_bai_elim.jl]() and [linear_topm_elim.jl]() for the experiments of Figure 1 (left and middle) and Table 3. They can be visualized with [plot_elim_algs.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/plot_elim_algs.jl), [plot_elim_full_vs_emp.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/plot_elim_full_vs_emp.jl), and [print_table_elim.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/print_table_elim.jl)
- [linear_bai_delta.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/linear_bai_delta.jl) for the experiment of Figure 1 (right). It can be visualized with [plot_delta.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/plot_delta.jl)
 
We also include a general visualization script [viz_experiment.jl](https://github.com/AndreaTirinzoni/bandit-elimination/blob/main/experiments/visualization/viz_experiment.jl) which takes as input a .dat file (produced by the run of some algorithm) and plots the corresponding sample complexities, computation times, and elimination times.

**Note**: reproducing the experiments can take a long time. Most files run at least 5 algorithms, each with 3 to 5 variants and each for 100 runs, which makes roughly 2000 runs per file. To get faster results, just edit the script file to use less repetitions and/or to run only some desired algorithm. If one only cares about reproducing the plots, our results can be found in the experiments/saved_results folder.

## References

[1] Degenne, R., Ménard, P., Shang, X., & Valko, M. (2020, November). _Gamification of pure exploration for linear bandits_. In International Conference on Machine Learning (pp. 2432-2442). PMLR.   
[2] Wang, P. A., Tzeng, R. C., & Proutiere, A. (2021). _Fast Pure Exploration via Frank-Wolfe_. Advances in Neural Information Processing Systems, 34.   
[3] Fiez, T., Jain, L., Jamieson, K. G., & Ratliff, L. (2019). _Sequential experimental design for transductive linear bandits_. Advances in neural information processing systems, 32.   
[4] Jedra, Y., & Proutiere, A. (2020). _Optimal best-arm identification in linear bandits_. Advances in Neural Information Processing Systems, 33, 10007-10017.   
[5] Soare, M., Lazaric, A., & Munos, R. (2014). _Best-arm identification in linear bandits_. Advances in Neural Information Processing Systems, 27.   
[6] Réda, C., Kaufmann, E., & Delahaye-Duriez, A. (2021, March). _Top-m identification for linear bandits_. In International Conference on Artificial Intelligence and Statistics (pp. 1108-1116). PMLR.    
[7] Réda, C., Tirinzoni, A., & Degenne, R. (2021). _Dealing With Misspecification In Fixed-Confidence Linear Top-m Identification_. Advances in Neural Information Processing Systems, 34.    
[8] Degenne, R., Koolen, W. M., & Ménard, P. (2019). _Non-asymptotic pure exploration by solving games_. Advances in Neural Information Processing Systems, 32.    
[9] Kalyanakrishnan, S., Tewari, A., Auer, P., & Stone, P. (2012, June). _PAC subset selection in stochastic multi-armed bandits_. In ICML (Vol. 12, pp. 655-662).    
[10] Gabillon, V., Ghavamzadeh, M., & Lazaric, A. (2012). _Best arm identification: A unified approach to fixed budget and fixed confidence_. Advances in Neural Information Processing Systems, 25.    
[11] Kaufmann, E., & Kalyanakrishnan, S. (2013, June). _Information complexity in bandit subset selection_. In Conference on Learning Theory (pp. 228-251). PMLR.
