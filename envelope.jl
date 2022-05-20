######################################################################################################
# Adapted functions from the implementation of Frank-Wolfe Sampling 
# (https://github.com/rctzeng/NeurIPS2021-Fast-Pure-Exploration-via-Frank-Wolfe)          
######################################################################################################
using JuMP;
import Tulip;

# adaptation of the function "hμ_in_lambda" from the original code
function μ_in_model(pep::Union{BAI, BAI_State}, μ, hi)
    for i=1:length(μ)
        if (i!=hi) && (μ[hi]-μ[i])<=eps()
            return false;
        end
    end
    return true;
end

function μ_in_model(pep::Union{Topm, Topm_State}, μ, hi)
    return true;
end

function μ_in_model(pep::Union{OSI, OSI_State}, μ, hi)
    for i=1:length(μ)
        if abs(μ[i]) <= eps()
            return false;
        end
    end
    return true;
end

function is_complete_square(n)
    p = floor(Int, sqrt(n));
    return p*p == n;
end

function solveZeroSumGame(M_payoff, K, n_row)
    # Note: our original code used with_optimizer() instead of optimizer_with_attributes(), but it seems that with_optimizer() is no longer available in later versions of the JuMP package
    m = Model(optimizer_with_attributes(Tulip.Optimizer));
    @variable(m, x[1:K] >= 0)
    @variable(m, w)
    for j in 1:n_row
        @constraint(m, sum(M_payoff[j][k]*x[k] for k=1:K) >= w)
    end
    @constraint(m, sum(x[i] for i=1:K) == 1)
    @objective(m, Max, w)
    optimize!(m);
    f_success = termination_status(m);
    z = JuMP.value.(x);
    return z;
end

"""
Linear BAI: computing f and ∇f by Proposition 1 of Wang et al. 2021
"""
function alt_min_envelope(pep::Union{BAI, BAI_State}, hw, θ_hat, Vxinv, use_elim::Bool)

    @assert !use_elim || typeof(pep) == BAI_State  "Must use a stateful pep with elimination"

    arms = pep.arms
    K = length(arms);
    hi = argmax([θ_hat'arms[k] for k=1:K]);
    active_arms = use_elim ? pep.active_arms : 1:K;
    suboptimal = [i for i in active_arms if i != hi];

    # construct ∇f
    f = zeros(length(suboptimal))
    ∇f = Vector{Float64}[]  # this will be num_suboptimal x K

    for (i,k) in enumerate(suboptimal)
        direction = arms[hi]-arms[k];
        λ = θ_hat - (direction'θ_hat / ((direction')*Vxinv*direction)) * Vxinv*(direction);
        push!(∇f, [((arms[j]')*(θ_hat-λ))^2 / 2 for j in 1:K])
        f[i] = hw'∇f[end]
    end

    return argmin(f), f, ∇f;
end

"""
Linear Top-m: computing f and ∇f by Proposition 1 of Wang et al. 2021
"""
function alt_min_envelope(pep::Union{Topm, Topm_State}, hw, θ_hat, Vxinv, use_elim::Bool)

    @assert !use_elim || typeof(pep) == Topm_State  "Must use a stateful pep with elimination"

    arms = pep.arms
    K = length(arms);

    topm = istar(pep, θ_hat)
    topm_active = use_elim ? setdiff(topm, pep.found_topm) : topm
    ∇f = Vector{Float64}[]  # this will be num_halfspaces x K
    f = Float64[]  # this will be num_halfspaces x 1

    for i in topm_active  # loop over (active) topm arms
        candidates = use_elim ? setdiff(1:K, topm, pep.worse_than[i]) : setdiff(1:K, topm)
        for j in candidates # loop over (active) "ambiguous arms"
            direction = arms[i]-arms[j];
            λ = θ_hat - (direction'θ_hat / ((direction')*Vxinv*direction)) * Vxinv*(direction);
            push!(∇f, [((arms[k]')*(θ_hat-λ))^2 / 2 for k in 1:K])
            push!(f, hw'∇f[end])
        end
    end

    return argmin(f), f, ∇f;
end

"""
Linear Threshold: computing f and ∇f by Proposition 1 of Wang et al. 2021s
"""
function alt_min_envelope(pep::Union{OSI, OSI_State}, hw, θ_hat, Vxinv, use_elim::Bool)

    @assert !use_elim || typeof(pep) == OSI_State  "Must use a stateful pep with elimination"

    K = length(pep.arms); 
    active_arms = use_elim ? [k for k in pep.active_arms] : 1:K

    # construct ∇f
    f = zeros(length(active_arms))
    ∇f = Vector{Float64}[]  # this will be num_active_arms x K

    for (i,k) in enumerate(active_arms)
        x = -(θ_hat')*pep.arms[k]; 
        ak = pep.arms[k];
        λ = θ_hat + sign(x) * (x/((ak')*Vxinv*ak)) * Vxinv * ak;
        push!(∇f, [((arms[j]')*(θ_hat-λ))^2 / 2 for j in 1:K])
        f[i] = hw'∇f[end]
    end

    return argmin(f), f, ∇f;
end

function compute_f_∇f(pep::Union{BAI, BAI_State, Topm, Topm_State, OSI, OSI_State}, hw, θ_hat, r, Vxinv, use_elim::Bool)

    @assert !use_elim || typeof(pep) == BAI_State || typeof(pep) == OSI_State || typeof(pep) == Topm_State  "Must use a stateful pep with elimination"

    fargmin, f, ∇f = alt_min_envelope(pep, hw, θ_hat, Vxinv, use_elim);
    fidx = r > eps() ? [j for j in 1:length(f) if f[j] < f[fargmin] + r] : [fargmin];
    
    return f, ∇f, fidx;
end

"""
Compute an optimal allocation by solving the lower bound optimization problem
"""
function optimal_allocation(pep::Union{BAI, BAI_State, OSI, OSI_State, Topm, Topm_State}, θ, use_elim::Bool, max_iter=1000)

    @assert !use_elim || typeof(pep) == BAI_State || typeof(pep) == OSI_State || typeof(pep) == Topm_State  "Must use a stateful pep with elimination"

    K = narms(pep)
    w = ones(K) ./ K
    for i=1:max_iter
        Vwinv = pinv(sum([w[k]*pep.arms[k]*(pep.arms[k]') for k=1:K]))
        _, ∇f, fidx = compute_f_∇f(pep, w, θ, 0, Vwinv, use_elim)
        w_next = zeros(K)
        w_next[argmax(∇f[fidx[1]])] = 1.0
        w = w*(i/(i+1)) + w_next/(i+1)
        if norm(w_next) / ((i+1)*norm(w)) < 0.001
            break
        end
    end
    return w
end