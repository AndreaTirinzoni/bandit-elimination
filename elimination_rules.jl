struct CompElim end # computationally-efficient elimination
long(elim_rule::CompElim) = "(C)";
abbrev(elim_rule::CompElim) = "C";

struct StatElim end # statistically-efficient elimination
long(elim_rule::StatElim) = "(S)";
abbrev(elim_rule::StatElim) = "S";

struct NoElim end # no elimination (this is just a placeholder to unify algorithms)
long(elim_rule::NoElim) = "";
abbrev(elim_rule::NoElim) = "";

function eliminate(pep::BAI_State, elim_rule::NoElim, β, t, δ, θ_hat, Vinv) end
function eliminate(pep::Topm_State, elim_rule::NoElim, β, t, δ, θ_hat, Vinv) end
function eliminate(pep::OSI_State, elim_rule::NoElim, β, t, δ, θ_hat, Vinv) end

# BAI - computationally efficient
function eliminate(pep::BAI_State, elim_rule::CompElim, β, t, δ, θ_hat, Vinv)
    to_elim = Set{Int32}([])
    arm_star = armstar(pep,θ_hat)
    for j in pep.active_arms
        inf_llr, _ = alternative(Vinv, θ_hat, arm_star, pep.arms[j])
        if inf_llr > β(t, δ)
            push!(to_elim, j)
            if pep.elim_times[j] == 0
                pep.elim_times[j] = t
            end
        end
    end
    setdiff!(pep.active_arms, to_elim)
end

# BAI - statistically efficient
function eliminate(pep::BAI_State, elim_rule::StatElim, β, t, δ, θ_hat, Vinv)
    to_elim = Set{Int32}([])
    for j in pep.active_arms
        for i in 1:length(pep.arms)  # TODO: order arms by mean and loop only over arms better than j
            inf_llr, _ = alternative(Vinv, θ_hat, pep.arms[i], pep.arms[j])
            if inf_llr > β(t, δ)
                push!(to_elim, j)
                if pep.elim_times[j] == 0
                    pep.elim_times[j] = t
                end
            end
        end
    end
    setdiff!(pep.active_arms, to_elim)
end

# Topm - computationally efficient
function eliminate(pep::Topm_State, elim_rule::CompElim, β, t, δ, θ_hat, Vinv)
    topm = istar(pep,θ_hat)
    for j in setdiff(topm, pep.found_topm)  # loop over topm arms of θ_hat which are still not labeled as top of θ
        for k in setdiff(1:narms(pep), topm, pep.worse_than[j])  # loop over all arms which are not topm and are still active for j
            inf_llr, _ = alternative(Vinv, θ_hat, pep.arms[j], pep.arms[k])
            if inf_llr > β(t, δ)
                push!(pep.worse_than[j], k)
            end
        end
        if length(pep.worse_than[j]) >= narms(pep) - pep.m
            push!(pep.found_topm, j)
        end
    end
end

# Topm - statistically efficient
function eliminate(pep::Topm_State, elim_rule::StatElim, β, t, δ, θ_hat, Vinv)
    for j in setdiff(1:narms(pep), pep.found_topm)  # loop over all arms which are still not labeled as top of θ
        for k in setdiff(1:narms(pep), [j], pep.worse_than[j])  # loop over all arms are still active for j
            inf_llr, _ = alternative(Vinv, θ_hat, pep.arms[j], pep.arms[k])
            if inf_llr > β(t, δ)
                push!(pep.worse_than[j], k)
            end
        end
        if length(pep.worse_than[j]) >= narms(pep) - pep.m
            push!(pep.found_topm, j)
        end
    end
end

# OSI - computationally efficient
function eliminate(pep::OSI_State, elim_rule::CompElim, β, t, δ, θ_hat, Vinv)
    to_elim = Set{Int32}([])
    signs = istar(pep,θ_hat)
    for j in pep.active_arms
        if signs[j] == 1
            arm1 = pep.arms[j]
            arm2 = zeros(length(arm1))
        else
            arm2 = pep.arms[j]
            arm1 = zeros(length(arm2))
        end
        inf_llr, _ = alternative(Vinv, θ_hat, arm1, arm2)
        if inf_llr > β(t, δ)
            push!(to_elim, j)
            pep.found_signs[j] = signs[j]
            if pep.elim_times[j] == 0
                pep.elim_times[j] = t
            end
        end
    end
    setdiff!(pep.active_arms, to_elim)
end

# OSI - statistically efficient (equivalent to comp. efficient)
function eliminate(pep::OSI_State, elim_rule::StatElim, β, t, δ, θ_hat, Vinv)
    return eliminate(pep, CompElim(), β, t, δ, θ_hat, Vinv)
end