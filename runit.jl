using Random;
using LinearAlgebra;

# Run the learning algorithm, parameterized by:
#   - a sampling rule from sampling_rules.jl
#   - a stopping rule from stopping_rules.jl (this also includes the recommendation rule)

function play!(k, rng, pep, θ, S, N, Vinv, R)
    ϕ = pep.arms[k]  # chosen feature vector
    Y = ϕ'θ + randn(rng);  # sample reward from N(ϕ'θ, 1)
    S .+= Y .* ϕ  # sum of rewards times features
    R[k] += Y   # sum of rewards for each arm
    Vinv .= sherman_morrison(Vinv, ϕ)  # update Vinv by sherman_morrison
    N[k] += 1  # update pull counts
end

function runit(seed, sampling, stopping, elim_rule::Union{NoElim,CompElim,StatElim}, θ, pep::Union{BAI,Topm,OSI}, β, δ, reset_active=false)
    # seed: random seed. UInt.
    # sampling: sampling rule.
    # stopping: stopping rule.
    # elim_rule: what type of elimination to use (NoElim for no elimination)
    # θ: true parameter.
    # pep: pure exploration problem.
    # β: threshold (could be used by both sampling and stopping rules).

    alg_name = long(sampling) * "+" * long(stopping) * long(elim_rule)
    println("Starting ", alg_name, " -- seed ", seed)

    rng = MersenneTwister(seed)

    # create the pep state (to keep track of active/eliminated stuff)
    pep = init_pep_state(pep)

    K = narms(pep)  # number of arms
    d = length(θ)  # parameter dimension
    N = zeros(Int64, K)  # pull counts
    S = zeros(d)  # sum of features times rewards
    R = zeros(K)  # sum of rewards for each arm
    Vinv = Matrix{Float64}(I, d, d)  # inverse of design matrix (initialized to identity matrix so that regularization = 1)

    start_time = time_ns()

    # pull each arm once. TODO: do we need this?
    for k = 1:K
        play!(k, rng, pep, θ, S, N, Vinv, R)
    end

    state = start(sampling, pep, N)

    t = sum(N)

    while true

        # least squares estimator
        θ_hat = Vinv * S

        # reset the sets of active arms if the sampling rule uses elimination
        if check_power2(t) && hasproperty(sampling, :ElimType) && reset_active
            reset_pep_state(pep)
        end

        # check eliminations
        eliminate(pep, elim_rule, β, t, δ, θ_hat, Vinv)

        # check stopping rule
        should_stop, answer = stop(stopping, pep, β, t, δ, θ_hat, Vinv)
        if should_stop
            println("End ", alg_name, " -- seed ", seed, " -- samples ", sum(N), " -- time ", (time_ns() - start_time)/1e9, " s\n")
            return (answer, copy(N), time_ns() - start_time, pep.elim_times)
        end

        t += 1

        # invoke sampling rule
        k, internal_stop, internal_answer = nextsample(state, pep, β, t, δ, θ_hat, N, S, Vinv, R)

        # for native elimination-based algorithm, check the internal stopping rule
        if internal_stop
            println("End ", alg_name, " -- seed ", seed, " -- samples ", sum(N), " -- time ", (time_ns() - start_time)/1e9, " s\n")
            return (internal_answer, copy(N), time_ns() - start_time)
        end

        # play arm
        play!(k, rng, pep, θ, S, N, Vinv, R)
    end
end
