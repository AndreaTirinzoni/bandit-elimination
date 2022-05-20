# Sampling rules.
# We organise them in two levels
# - sampling rule; a factory for sampling rule states
# - sampling rule state; keeps track of i.e. tracking information etc.

include("regret.jl");
include("tracking.jl");

"""
Combining sampling rules with elimination
"""

struct ElimSR end  # used to indicate that a sampling rule is combined with elimination
struct NoElimSR end  # same but for no elimination

abbrev(_::Type{ElimSR}) = "(E)";
abbrev(_::Type{NoElimSR}) = "";


"""
Uniform sampling
"""

struct RoundRobin end

long(sr::RoundRobin) = "Uniform";
abbrev(sr::RoundRobin) = "U";

function start(sr::RoundRobin, pep::Union{BAI_State, Topm_State, OSI_State}, N)
    return sr
end

function nextsample(sr::RoundRobin, pep, β, t, δ, θ_hat, N, S, Vinv, R)
    return (1 + (sum(N) % length(N))), false, Nothing
end


"""
Tracking fixed weights
"""

struct FixedWeights # used as factory and state
    w
    function FixedWeights(w)
        @assert all(w .≥ 0) && sum(w) ≈ 1 "$w not in simplex"
        new(w)
    end
end

long(sr::FixedWeights) = "Oracle";
abbrev(sr::FixedWeights) = "LBO";

function start(sr::FixedWeights, pep::Union{BAI_State, Topm_State, OSI_State}, N)
    return sr
end

function nextsample(sr::FixedWeights, pep, β, t, δ, θ_hat, N, S, Vinv, R)
    argmin(N .- sum(N) .* sr.w), false, Nothing
end


"""
LinGame (with a single learner as in MisLID)
"""

struct LinGame
    TrackingRule
    ElimType
    use_optimism::Bool
end

long(sr::LinGame) = "LinGame" * abbrev(sr.ElimType);
abbrev(sr::LinGame) = "LG" * abbrev(sr.ElimType);

struct LinGameState
    learner  # single online learner
    tr  # tracking rule
    elim_type  # type of elimination used (ElimSR or NoElimSR)
    use_optimism::Bool  # whether to use optimistic or greedy gains
    LinGameState(TrackingRule, N, ElimType, use_optimism) = new(
        AdaHedge(length(N)),
        TrackingRule(vec(N)),
        ElimType(),
        use_optimism
    )
end

function start(sr::LinGame, pep::Union{BAI_State, Topm_State, OSI_State}, N)
    LinGameState(sr.TrackingRule, N, sr.ElimType, sr.use_optimism)
end

# optimistic gradients
function optimistic_gradient(pep, t, θ_hat, λ, Vinv, use_optimism)
    K = length(pep.arms)
    grads = zeros(K)
    for k = 1:K
        arm = pep.arms[k]
        ref_value = (θ_hat .- λ)'arm
        confidence_width = use_optimism ? log(t) : 0
        deviation = sqrt(2 * confidence_width * ((arm') * Vinv * arm))
        ref_value > 0 ? grads[k] = 0.5 * (ref_value + deviation)^2 : grads[k] = 0.5 * (ref_value - deviation)^2
        #grads[k] = min(grads[k], confidence_width)  # TODO: check this clipping
    end
    return grads
end

function nextsample(sr::LinGameState, pep, β, t, δ, θ_hat, N, S, Vinv, R)

    # query the learner
    w = act(sr.learner)

    # Compute design matrix with w
    d = length(pep.arms[1])
    Vinv_w = zeros(d,d)
    for k in 1:narms(pep)
        Vinv_w .+= w[k] .* (pep.arms[k]*transpose(pep.arms[k]))
    end
    Vinv_w = inv(Vinv_w)
    
    # compute closest alternative
    _, λ = typeof(sr.elim_type) == ElimSR ? min_alt_active(pep, θ_hat, Vinv_w) : min_alt(pep, θ_hat, Vinv_w)

    # get optimistic gain
    ∇ = optimistic_gradient(pep, t, θ_hat, λ, Vinv, sr.use_optimism)

    incur!(sr.learner, -∇)

    # tracking
    k = track(sr.tr, vec(N), w)
    
    return k, false, Nothing
end

"""
LinGapE (Xu et al. 2018)
"""

struct LinGapE 
    ElimType
end

long(sr::LinGapE) = "LinGapE" * abbrev(sr.ElimType);
abbrev(sr::LinGapE) = "LGE" * abbrev(sr.ElimType);

function start(sr::LinGapE, pep::Union{BAI_State, Topm_State, OSI_State}, N)
    return sr
end

function gap(arm1, arm2, θ)
    (arm1 - arm2)'θ
end

function confidence(arm1, arm2, Vinv)
    sqrt(transpose(arm1 - arm2) * Vinv * (arm1 - arm2))
end

function nextsample(sr::LinGapE, pep::Union{BAI_State, Topm_State, OSI_State}, β, t, δ, θ_hat, N, S, Vinv, R)
    
    K = narms(pep)
    d = length(pep.arms[1])
    # c_t = sqrt(2 * β(t,δ))
    c_t = sqrt(2 * (log(1/δ) + 2*log(t+1)))  # heuristic threshold

    if typeof(pep) == BAI_State # Original LinGapE
        # Compute arms i_t and j_t as in Algorithm 2 of Xu et al.
        i_t = istar(pep, θ_hat)
        # Extension to elimination: to compute the most ambiguous arm j_t we just loop over active arms
        candidates = sr.ElimType == ElimSR ? pep.active_arms : 1:K
        # TODO check the following line
        # To find the arm maximizing the LinGapE "score" among candidates, 
        # we compute the maximum value of couples (j,score(j)) and pick its second element
        j_t = maximum([(gap(pep.arms[j], pep.arms[i_t], θ_hat) + 
                        confidence(pep.arms[j], pep.arms[i_t], Vinv) * c_t, j) for j in candidates])[2]                 
    elseif typeof(pep) == Topm_State # m-LinGapE: Extension to Topm proposed by Reda et al. 2021
        topm = istar(pep, θ_hat)
        topm_active = sr.ElimType == ElimSR ? setdiff(topm, pep.found_topm) : topm
        max_val = -Inf
        i_t = Nothing
        j_t = Nothing
        for i in topm_active  # loop over (active) topm arms
            candidates = sr.ElimType == ElimSR ? setdiff(1:K, topm, pep.worse_than[i]) : setdiff(1:K, topm)
            for j in candidates # loop over (active) "ambiguous arms"
                val = gap(pep.arms[j], pep.arms[i], θ_hat) + confidence(pep.arms[j], pep.arms[i], Vinv) * c_t
                if val > max_val
                    max_val = val
                    i_t = i
                    j_t = j
                end
            end
        end
    elseif typeof(pep) == OSI_State  # TODO check this
        zero_vec = zeros(length(pep.arms[1]))
        candidates = sr.ElimType == ElimSR ? pep.active_arms : 1:K
        j_t = maximum([(-abs(gap(pep.arms[j], zero_vec, θ_hat)) + 
                        confidence(pep.arms[j], zero_vec, Vinv) * c_t, j) for j in candidates])[2]  
    else
        @assert false
    end

    leading_arm = typeof(pep) == OSI_State ? zero_vec : pep.arms[i_t]
    ambiguous_arm = pep.arms[j_t]
    k = argmin([confidence(
        leading_arm,
        ambiguous_arm,
        sherman_morrison(Vinv, pep.arms[i]),
    ) for i = 1:K])

    return k, false, Nothing
end

"""
Frank-Wolfe Sampling (Wang et al. 2021)
"""
struct FWSampling
    ElimType
end

long(sr::FWSampling) = "FW-Sampling" * abbrev(sr.ElimType);
abbrev(sr::FWSampling) = "FWS" * abbrev(sr.ElimType);

mutable struct FWSamplingState
    x;
    Vxinv;
    ElimType;
    FWSamplingState(N, Vxinv, ElimType) = new(Float64.(N)./sum(N), Vxinv, ElimType);
end

function start(sr::FWSampling, pep::Union{BAI_State, Topm_State, OSI_State}, N)
    # compute Vxinv first
    dim = length(pep.arms[1])
    Vxinv = zeros(Float64, dim, dim);
    for k in 1:K
        Vxinv += pep.arms[k]*(pep.arms[k]')*N[k];
    end
    FWSamplingState(N, Vxinv./sum(N), sr.ElimType);
end

function nextsample(sr::FWSamplingState, pep::Union{BAI_State, Topm_State, OSI_State}, β, t, δ, θ_hat, N, S, Vinv, R)
    K = length(N);
    μ_hat = [θ_hat'pep.arms[k] for k=1:K]; 
    hi = argmax(μ_hat);

    r = t^(-9/10)/K; 
    z = [0.0 for i=1:K];

    if !μ_in_model(pep, μ_hat, hi) || is_complete_square(floor(Int, t/K))
        z = [1.0/K for i=1:K];
    else
        f, ∇f, fidx = compute_f_∇f(pep, sr.x, θ_hat, r, sr.Vxinv, sr.ElimType == ElimSR ? true : false);
        if length(fidx) == 1 # best challenger
            challenger_idx = argmax(∇f[fidx[1]]);
            z = [(challenger_idx==j) ? 1 : 0 for j=1:K];
        else # solve LP of the zero-sum matrix game
            Σ = [[(i==j) ? 1 : 0 for j=1:K]-sr.x for i=1:K];
            A = [[Σ[i]'∇f[j] for i=1:K] for j in fidx]; # construct payoff matrix
            z = solveZeroSumGame(A, K, length(fidx));
        end
    end
    setfield!(sr, :x, sr.x*((t-1.0)/t) + z*1.0/t);

    # Incremental update of inverse design matrix
    # nextVxinv = sherman_morrison(sr.Vxinv*(t-1.0)/t, z[1]*pep.arms[1]/t);
    # for k=2:K
    #     nextVxinv = sherman_morrison(nextVxinv, z[k]*pep.arms[k]/t);
    # end
    # setfield!(sr, :Vxinv, nextVxinv);

    # Direct computation of inverse design matrix (TODO: this seems more efficient for many arms and small d)
    d = length(pep.arms[1])
    Vinv_w = zeros(d,d)
    for k in 1:narms(pep)
        Vinv_w .+= sr.x[k] .* (pep.arms[k]*transpose(pep.arms[k]))
    end
    Vinv_w = inv(Vinv_w)
    setfield!(sr, :Vxinv, Vinv_w);

    return argmax(sr.x ./ N), false, Nothing
end

"""
Lazy T&S (Jedra and Proutiere, 2020)
Implementation of (https://github.com/rctzeng/NeurIPS2021-Fast-Pure-Exploration-via-Frank-Wolfe)  
"""
struct LazyTaS
    ElimType
end

long(sr::LazyTaS) = "LazyTaS" * abbrev(sr.ElimType);
abbrev(sr::LazyTaS) = "LTS" * abbrev(sr.ElimType);

mutable struct LazyTaSState
    sumw;
    w;
    A;
    A0;
    c0;
    i0;
    ElimType;
    LazyTaSState(N, ElimType, A, A0, c0, i0) = new(zeros(length(N)), [1.0/length(N) for i=1:length(N)], A, A0, c0, i0, ElimType);
end

function start(sr::LazyTaS, pep::Union{BAI_State, Topm_State, OSI_State}, N)

    dim = length(pep.arms[1])
    # Compute spanner
    A = zeros(dim, dim)
    A0 = zeros(Int64, dim);
    r = 0;
    k = 1
    while r < dim
        if rank(A + pep.arms[k]*(pep.arms[k]')) > r
            A += pep.arms[k]*(pep.arms[k]');
            A0[r+1] = k;
            r += 1;
        end
        k += 1
    end
    c0 = minimum(eigvals(A)) / sqrt(dim);
    i0 = 0;
    # TODO: defined in the original code but not used later 
    #c1 = minimum(eigvals(A));
    #c2 = 1.1; # (1+u) * (sigma^2), where u=0.1, sigma=1
    #c3 = dim*log(sqrt(11)) # dim * log(sqrt(u^-1 + 1)), used by DesignType="Heuristic"

    LazyTaSState(N, sr.ElimType, A, A0, c0, i0);
end

function nextsample(sr::LazyTaSState, pep::Union{BAI_State, Topm_State, OSI_State}, β, t, δ, θ_hat, N, S, Vinv, R)
    K = narms(pep); 
    dim = length(pep.arms[1]);

    if check_power2(t) # lazy update
        w = copy(sr.w);
        for i=1:1000 # the setting in Lazy T&S (Jedra and Proutiere, 2020)
            Vwinv = pinv(sum([w[k]*pep.arms[k]*(pep.arms[k]') for k=1:K]));
            _, ∇f, fidx = compute_f_∇f(pep, w, θ_hat, 0, Vwinv, sr.ElimType == ElimSR ? true : false);
            w_next = zeros(K);
            w_next[argmax(∇f[fidx[1]])] = 1.0;
            w = w*(i/(i+1)) + w_next/(i+1);
            if norm(w_next) / ((i+1)*norm(w)) < 0.001
                break
            end
        end
        setfield!(sr, :w, w);
    end
    setfield!(sr, :sumw, sr.sumw+sr.w);
     # we simplify the arm tracking rule, without implementing their special design for coping issues arised in "many-arm" setting
    # TODO check: we replace minimum(eigvals(sr.A)) with 1 / maximum(eigvals(Vinv))
    (1 / maximum(eigvals(Vinv)) < sr.c0 * sqrt(t)) ? arm = sr.A0[sr.i0+1] : arm = argmin(N - sr.sumw);

    setfield!(sr, :i0, (sr.i0+1) % dim);

    return arm, false, Nothing
end

"""
XYAdaptive (Soare et al., 2014) 
"""
struct XYAdaptive end

long(sr::XYAdaptive) = "XY-Adaptive";
abbrev(sr::XYAdaptive) = "XYA";

mutable struct XYAdaptiveState
    ρ
    ρ_old
    t_old
    Xactive
    α
    update_after  # recompute the arm to play only after some steps (for efficiency reasons)
    last_arm  # the last played arm (to be repeated for update_after steps)
    remaining_steps  # how many steps the last_arm should still be played
    XYAdaptiveState(ρ, ρ_old, t_old, Xactive, α) = new(ρ, ρ_old, t_old, Xactive, α, 10, 0, 0);
end

function start(sr::XYAdaptive, pep::BAI_State, N)
    ρ = 1
    ρ_old = 1
    t_old = sum(N)
    Xactive = copy(pep.arms)
    α = 0.1
    return XYAdaptiveState(ρ, ρ_old, t_old, Xactive, α)
end

function drop_arms(Xactive, Vinv, θ_hat, β)
    X = copy(Xactive)
    K = length(Xactive)
    for i = 1:K
        arm = X[i]
        for j = 1:K
            if j == i
                continue
            end
            arm_prime = X[j]
            y = arm_prime - arm
            if (y' * Vinv * y * 2 * β)^0.5 <= y'θ_hat
                filter!(x -> x ≠ arm, Xactive)
                break
            end
        end
    end
    return Xactive
end

function build_gaps(arms)
    gaps = Vector{Float64}[]
    for pair in subsets(arms, 2)
        gap1 = pair[1] - pair[2]
        push!(gaps, gap1)
        # gap2 = pair[2] - pair[1]
        # push!(gaps, gap2)
    end
    return gaps
end

function nextsample(sr::XYAdaptiveState, pep::BAI_State, β, t, δ, θ_hat, N, S, Vinv, R)

    if sr.remaining_steps == 0

        nb_I = nanswers(pep)

        Y = build_gaps(sr.Xactive)
        nb_gaps = length(Y)
        vals = [maximum([transpose(Y[i]) * sherman_morrison(Vinv, pep.arms[j]) * Y[i] for i = 1:nb_gaps]) for j = 1:nb_I]
        k = randmin(vals)
        setfield!(sr, :last_arm, k)
        setfield!(sr, :remaining_steps, sr.update_after)

        if sr.ρ / t < sr.α * sr.ρ_old / sr.t_old
            setfield!(sr, :t_old, t);
            setfield!(sr, :ρ_old, sr.ρ);
            Xcopy = copy(pep.arms)
            Xactive_new = drop_arms(Xcopy, Vinv, θ_hat, β(t, δ))
            setfield!(sr, :Xactive, Xactive_new);
        end

        # ρ = maximum([transpose(Y[i]) * sherman_morrison(Vinv, pep.arms[k]) * Y[i] for i = 1:nb_gaps])
        ρ = vals[k]
        setfield!(sr, :ρ, ρ);

    end

    setfield!(sr, :remaining_steps, sr.remaining_steps - 1);
    k = sr.last_arm

    if length(sr.Xactive) <= 1
        stop = true
        answer = length(sr.Xactive) == 1 ? findall(x -> x == sr.Xactive[1], pep.arms)[1] : rand(1:narms(pep))
    else
        stop = false
        answer = Nothing
    end

    return k, stop, answer
end


"""
RAGE (Fiez et al. 2019)
"""
struct RAGE end

long(sr::RAGE) = "RAGE";
abbrev(sr::RAGE) = "RG";

mutable struct RAGEState
    active_arms
    phase_index
    factor
    allocation
    RAGEState(active_arms,factor) = new(active_arms, 1, factor, Nothing);
end

function start(sr::RAGE, pep::BAI_State, N)
    active_arms = [k for k in 1:narms(pep)]
    factor = 10.0 # from Fiez's code
    return RAGEState(active_arms, factor)
end

function build_Y(active_arms, arms)

    K = length(active_arms)
    d = length(arms[1])
    Zhat = zeros(K, d)
    for i in 1:K
        Zhat[i, :] = arms[active_arms[i]]
    end
    Y = zeros(K*K, d)
    
    for i in 0:K-1
        Y[K*i+1:K*(i+1), :] = Zhat .- reshape(Zhat[i+1, :], 1, d)
    end
    
    return Y
end

function optimal_allocation(arms, Y)

    K = length(arms)
    d = length(arms[1])
    design = ones(K) ./ K
    rho = 0

    arms_vec = zeros(K, d)
    for k in 1:K
        arms_vec[k, :] = arms[k]
    end
    
    max_iter = 5000
    
    for count in 1:max_iter

        A_inv = zeros(d,d)
        for k in 1:K
            A_inv .+= design[k] .* (arms[k]*transpose(arms[k]))
        end
        A_inv = pinv(A_inv)
    
        U, D, V = svd(A_inv)
        Ainvhalf = U * Diagonal(sqrt.(D)) * V
        
        newY = (Y * Ainvhalf).^2
        rho = sum(newY, dims=2)[:, 1]
        
        idx = argmax(rho)
        y = Y[idx, :]
        g = vec((arms_vec * A_inv * y).^2)
        g_idx = argmax(g)
                    
        gamma = 2/(count+2)
        design_update = copy(design)
        design_update .*= -gamma
        design_update[g_idx] += gamma
            
        relative = norm(design_update)/(norm(design))
                    
        design .+= design_update
                        
        if relative < 0.01
             break
        end
    end
                    
    design[design .< 1e-5] .= 0
    
    return design, maximum(rho)

end

function rounding(design, num_samples)

    support_idx = findall(x -> x > 0, design)
    num_support = length(support_idx)
    support = design[support_idx]
    n_round = ceil.((num_samples - 0.5*num_support).*support)

    while sum(n_round) - num_samples != 0
        if sum(n_round) < num_samples
            idx = argmin(n_round ./ support)
            n_round[idx] += 1
        else
            idx = argmax((n_round .- 1) ./ support)
            n_round[idx] -= 1
        end
    end

    allocation = zeros(length(design))
    allocation[support_idx] = n_round
    
    return Int.(allocation)

end

function drop_arms_rage(active_arms, arms, Vinv, θ_hat, δ_t)

    new_active = copy(active_arms)

    for arm_idx in active_arms

        arm = arms[arm_idx]

        for arm_idx_prime in active_arms

            if arm_idx == arm_idx_prime
                continue
            end

            arm_prime = arms[arm_idx_prime]
            y = arm_prime - arm

            if (2 * y' * Vinv * y * log(2*length(arms)^2/δ_t))^0.5 <= y'θ_hat
                filter!(x -> x ≠ arm_idx, new_active)
                break
            end
        end
    end

    return new_active
end

function nextsample(sr::RAGEState, pep::BAI_State, β, t, δ, θ_hat, N, S, Vinv, R)

    # change of phase
    if sr.allocation == Nothing || sum(sr.allocation) == 0

        δ_t = δ / sr.phase_index ^ 2

        # update set of active arms after the end of the previous phase
        if sr.allocation != Nothing
            new_active = drop_arms_rage(sr.active_arms, pep.arms, Vinv, θ_hat, δ_t)
            setfield!(sr, :active_arms, new_active);
            setfield!(sr, :phase_index, sr.phase_index + 1);
        end

        K = narms(pep)     
                
        Y = build_Y(sr.active_arms, pep.arms)
        design, rho = optimal_allocation(pep.arms, Y)
        support = count(x -> x > 0, design)
        n_min = 2 * sr.factor * support
        eps = 1 / sr.factor
        
        num_samples = Int(max(ceil(8*(2^(sr.phase_index-1))^2*rho*(1+eps)*log(2*K^2/δ_t)), n_min))
        
        allocation = rounding(design, num_samples)
        setfield!(sr, :allocation, allocation);
    end
    
    # play one arm in the allocation
    arm = argmax(sr.allocation)
    sr.allocation[arm] -= 1 

    # check stopping
    if length(sr.active_arms) <= 1
        stop = true
        answer = length(sr.active_arms) == 1 ? first(sr.active_arms) : rand(1:narms(pep))
    else
        stop = false
        answer = Nothing
    end

    return arm, stop, answer

end

"""
LinGIFA (Reda et al. 2021)
"""

struct LinGIFA
end

long(sr::LinGIFA) = "LinGIFA";
abbrev(sr::LinGIFA) = "LGF";

function start(sr::LinGIFA, pep::Union{BAI_State, Topm_State, OSI_State}, N)
    return sr
end

function nextsample(sr::LinGIFA, pep::Union{BAI_State, Topm_State}, β, t, δ, θ_hat, N, S, Vinv, R)
    
    K = narms(pep)
    conf_t = sqrt(2 * β(t,δ))

    # compute B_{i,j}(t)
    B_t = zeros(K,K)
    for i in 1:K
        for j in 1:K
            if i == j
                continue
            end
            B_t[i,j] = gap(pep.arms[i], pep.arms[j], θ_hat) + confidence(pep.arms[j], pep.arms[i], Vinv) * conf_t
        end
    end

    if typeof(pep) == BAI_State
        b_t = 0
        c_t = 0
        min_val = Inf
        for j in 1:K
            vals = B_t[:, j]
            vals[j] = -Inf
            idx = argmax(vals)
            if vals[idx] < min_val
                min_val = vals[idx]
                b_t = j
                c_t = idx
            end
        end
                       
    elseif typeof(pep) == Topm_State

        max_m_B_t = zeros(K)
        for j in 1:K
            vals = B_t[:, j]
            vals[j] = -Inf
            idx = partialsortperm(vals, pep.m, rev=true)
            max_m_B_t[j] = vals[idx]
        end

        # compute j_t
        J_t = partialsortperm(max_m_B_t, 1:pep.m)
        not_J_t = [i for i in 1:K if !(i in J_t)]

        # compute b_t and c_t
        b_t = J_t[argmax([maximum([B_t[i, j] for i in not_J_t]) for j in J_t])]
        c_t = not_J_t[argmax([B_t[i, b_t] for i in not_J_t])]
    end

    leading_arm = pep.arms[b_t]
    ambiguous_arm = pep.arms[c_t]
    k = argmin([confidence(
        leading_arm,
        ambiguous_arm,
        sherman_morrison(Vinv, pep.arms[i]),
    ) for i = 1:K])

    return k, false, Nothing
end

"""
LUCB (Kalyanakrishnan et al. 2012)
"""

struct LUCB 
    ElimType
end

long(sr::LUCB) = "LUCB" * abbrev(sr.ElimType);
abbrev(sr::LUCB) = "LUCB" * abbrev(sr.ElimType);

function start(sr::LUCB, pep::Union{BAI_State, Topm_State, OSI_State}, N)
    return sr
end

function nextsample(sr::LUCB, pep::Union{BAI_State, Topm_State, OSI_State}, β, t, δ, θ_hat, N, S, Vinv, R)

    K = narms(pep)
    μ_hat = R ./ N  # unstructured mean estimator
    conf_t = sqrt.(2 * β(t,δ) ./ N)   # unstructured confidence interval (simplified)
    lconf = μ_hat .- conf_t
    uconf = μ_hat .+ conf_t

    if typeof(pep) == BAI_State
        b_t = argmax(μ_hat)
        candidates = sr.ElimType == ElimSR ? pep.active_arms : 1:K
        c_t = maximum([(uconf[j] - lconf[b_t], j) for j in candidates if j != b_t])[2]
        # check native stopping rule
        if uconf[c_t] - lconf[b_t] <= 0
            stop = true
            answer = b_t
        else
            stop = false
            answer = Nothing
        end
    elseif typeof(pep) == Topm_State
        topm = sortperm(μ_hat, rev=true)[1:pep.m]
        topm_active = sr.ElimType == ElimSR ? setdiff(topm, pep.found_topm) : topm
        max_val = -Inf
        b_t = Nothing
        c_t = Nothing
        for i in topm_active  # loop over (active) topm arms
            candidates = sr.ElimType == ElimSR ? setdiff(1:K, topm, pep.worse_than[i]) : setdiff(1:K, topm)
            for j in candidates # loop over (active) "ambiguous arms"
                val = uconf[j] - lconf[i]
                if val > max_val
                    max_val = val
                    b_t = i
                    c_t = j
                end
            end
        end
        # check native stopping rule
        if uconf[c_t] - lconf[b_t] <= 0
            stop = true
            answer = topm
        else
            stop = false
            answer = Nothing
        end
    elseif typeof(pep) == OSI_State
        candidates = sr.ElimType == ElimSR ? pep.active_arms : 1:K
        c_t = maximum([(-abs(μ_hat[j]) + conf_t[j], j) for j in candidates])[2]
        b_t = c_t  # just to make sure c_t is played
        # check native stopping rule
        if -abs(μ_hat[c_t]) + conf_t[c_t] <= 0
            stop = true
            answer = [μ_hat[i] >= 0 ? 1 : 0 for i in 1:K]
        else
            stop = false
            answer = Nothing
        end
    else
        @assert false
    end

    # largest variance selection rule
    k = [c_t, b_t][argmin([N[c_t], N[b_t]])]

    return k, stop, answer
end

"""
UGapE (Gabillon et al. 2012)
"""

struct UGapE
end

long(sr::UGapE) = "UGapE";
abbrev(sr::UGapE) = "UGE";

function start(sr::UGapE, pep::Union{BAI_State, Topm_State, OSI_State}, N)
    return sr
end

function nextsample(sr::UGapE, pep::Union{BAI_State, Topm_State}, β, t, δ, θ_hat, N, S, Vinv, R)
    
    K = narms(pep)
    μ_hat = R ./ N  # unstructured mean estimator
    conf_t = sqrt.(2 * β(t,δ) ./ N)   # unstructured confidence interval (simplified)
    lconf = μ_hat .- conf_t
    uconf = μ_hat .+ conf_t

    # compute B_{i,j}(t)
    B_t = zeros(K,K)
    for i in 1:K
        for j in 1:K
            B_t[i,j] = uconf[i] - lconf[j]
        end
    end

    if typeof(pep) == BAI_State
        b_t = 0
        c_t = 0
        min_val = Inf
        for j in 1:K
            vals = B_t[:, j]
            vals[j] = -Inf
            idx = argmax(vals)
            if vals[idx] < min_val
                min_val = vals[idx]
                b_t = j
                c_t = idx
            end
        end   
        
        # check native stopping rule
        if B_t[c_t, b_t] <= 0
            stop = true
            answer = argmax(μ_hat)
        else
            stop = false
            answer = Nothing
        end
    elseif typeof(pep) == Topm_State
        max_m_B_t = zeros(K)
        for j in 1:K
            vals = B_t[:, j]
            vals[j] = -Inf
            idx = partialsortperm(vals, pep.m, rev=true)
            max_m_B_t[j] = vals[idx]
        end

        # compute j_t
        J_t = partialsortperm(max_m_B_t, 1:pep.m)
        not_J_t = [i for i in 1:K if !(i in J_t)]

        # compute b_t and c_t
        b_t = J_t[argmax([maximum([B_t[i, j] for i in not_J_t]) for j in J_t])]
        c_t = not_J_t[argmax([B_t[i, b_t] for i in not_J_t])]

        # check native stopping rule
        if maximum([max_m_B_t[j] for j in J_t]) <= 0
            stop = true
            answer = sortperm(μ_hat, rev=true)[1:pep.m]
        else
            stop = false
            answer = Nothing
        end
    end

    # largest variance selection rule
    k = [c_t, b_t][argmin([N[c_t], N[b_t]])]

    return k, stop, answer
end

"""
Racing algorithm (Kaufmann and Kalyanakrishnan, 2013)
"""

struct Racing
end

long(sr::Racing) = "Racing"
abbrev(sr::Racing) = "Rac"

mutable struct RacingState
    active_arms
    selected_arms
    discarded_arms
    allocation
    RacingState(active_arms) = new(active_arms, Int64[], Int64[], Nothing);
end

function start(sr::Racing, pep::Union{BAI_State, Topm_State}, N)
    active_arms = [k for k in 1:narms(pep)]
    return RacingState(active_arms)
end

function nextsample(sr::RacingState, pep::Union{BAI_State, Topm_State}, β, t, δ, θ_hat, N, S, Vinv, R)

    K = narms(pep)
    m = typeof(pep) == Topm_State ? pep.m : 1
    μ_hat = R ./ N  # unstructured mean estimator
    conf_t = sqrt.(2 * β(t,δ) ./ N)   # unstructured confidence interval (simplified)
    lconf = μ_hat .- conf_t
    uconf = μ_hat .+ conf_t

    # change of phase
    if sr.allocation == Nothing || sum(sr.allocation) == 0

        topm = sortperm(μ_hat, rev=true)[1:m]
        # Compute set of top-m arms and its complement
        J_t = setdiff(topm, sr.selected_arms)
        not_J_t = [k for k in sr.active_arms if !(k in J_t)]
        # Compute two candidate arms
        u_t = not_J_t[argmax([uconf[k] for k in not_J_t])]
        l_t = J_t[argmin([lconf[k] for k in J_t])]
        # Compute empirical best and worst active arms
        a_B = sr.active_arms[argmax([μ_hat[k] for k in sr.active_arms])]
        a_W = sr.active_arms[argmin([μ_hat[k] for k in sr.active_arms])]
        # Elimination and selection
        diff_aB = uconf[u_t] - lconf[a_B]
        diff_aW = uconf[a_W] - lconf[l_t]
        if diff_aB <= 0 && diff_aB <= 0
            a = [a_B, a_W][argmax([diff_aB, diff_aW])]
        elseif diff_aB <= 0
            a = a_B
        elseif diff_aW <= 0
            a = a_W
        else
            a = Nothing
        end

        if a != Nothing
            filter!(x -> x ≠ a, sr.active_arms)
            if a == a_B
                push!(sr.selected_arms, a)
            else
                push!(sr.discarded_arms, a)
            end
        end

        # new allocation plays each active arm once
        allocation = zeros(K)
        allocation[sr.active_arms] .= 1
        setfield!(sr, :allocation, allocation);
    end

    # play one arm in the allocation
    arm = argmax(sr.allocation)
    sr.allocation[arm] -= 1

    # check stopping
    if length(sr.selected_arms) >= m
        stop = true
        answer = m == 1 ? sr.selected_arms[1] : sr.selected_arms[1:m]
    elseif length(sr.discarded_arms) >= K - m
        stop = true
        answer = first(union(Set(sr.active_arms), Set(sr.selected_arms)), m)
    else
        stop = false
        answer = Nothing
    end

    return arm, stop, answer
end