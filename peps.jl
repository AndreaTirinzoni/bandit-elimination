# Pure exploration problems, each represented by:
#   - a basic structure recording static information (eg the set of arms)
#   - a state structure recording dynamic quantities (like active pieces)
# This implementation allows to easily share elimination sets among stopping and sampling rules

#################################################################
# Best-arm identification (BAI)
#################################################################

struct BAI
    arms;  # array of size K of arms in R^d
end

struct BAI_State
    arms;  # array of size K of arms in R^d
    active_arms::Set;  # set of arm indexes (from 1 to K) that are still active
    elim_times::Array;
    function BAI_State(arms)
        new(arms, Set(1:length(arms)), zeros(length(arms)))
    end
end

# factory for creating pep states
init_pep_state(pep::BAI) = BAI_State(pep.arms)

# reset active arms
function reset_pep_state(pep::BAI_State)
    union!(pep.active_arms, Set(1:length(arms)))
end

nanswers(pep::Union{BAI, BAI_State}) = length(pep.arms);
narms(pep::Union{BAI, BAI_State}) = length(pep.arms);
nactive(pep::BAI_State) = length(pep.active_arms);
istar(pep::Union{BAI, BAI_State}, θ) = argmax([arm'θ for arm in pep.arms]);
armstar(pep::Union{BAI, BAI_State}, θ) = pep.arms[argmax([arm'θ for arm in pep.arms])];
is_correct(pep::Union{BAI, BAI_State}, θ, answer) = answer == istar(pep, θ)

# closest alternative and value for all pieces
function min_alt(pep::Union{BAI, BAI_State}, θ, Vinv)
    opt_arm = armstar(pep, θ)
    opt_idx = istar(pep, θ)
    return minimum(alternative(Vinv, θ, opt_arm, pep.arms[k]) for k in 1:narms(pep) if k != opt_idx)
end

# closest alternative and value for active pieces only
function min_alt_active(pep::BAI_State, θ, Vinv)
    opt_arm = armstar(pep, θ)
    opt_idx = istar(pep, θ)
    return minimum(alternative(Vinv, θ, opt_arm, pep.arms[k]) for k in pep.active_arms if k != opt_idx)
end

#################################################################
# Top-m identification (Topm)
#################################################################

struct Topm
    arms;  # array of size K of arms in R^d
    m;  # number of best arms to identify
end

struct Topm_State
    arms;  # array of size K of arms in R^d
    m;  # number of best arms to identify
    worse_than::Dict;  # dictionary that maps arm indexes to sets of arm indexes which are found to have lower mean
    found_topm::Set;  # arm indexes found to be among the topm (updated incrementally)
    elim_times::Array;
    function Topm_State(arms, m)
        worse_than = Dict([(k, Set{Int32}()) for k in 1:length(arms)])
        found_topm = Set{Int32}()
        new(arms, m, worse_than, found_topm, zeros(length(arms)))
    end
end

# factory for creating pep states
init_pep_state(pep::Topm) = Topm_State(pep.arms, pep.m)

# reset active arms
function reset_pep_state(pep::Topm_State)
    empty!(pep.found_topm)
    for k in 1:length(pep.arms)
        empty!(pep.worse_than[k])
    end
end

nanswers(pep::Union{Topm, Topm_State}) = binomial(narms(pep), pep.m)
narms(pep::Union{Topm, Topm_State}) = length(pep.arms);
nfound(pep::Topm_State) = length(pep.found_topm);
istar(pep::Union{Topm, Topm_State}, θ) = sortperm([arm'θ for arm in pep.arms], rev=true)[1:pep.m];
is_correct(pep::Union{Topm, Topm_State}, θ, answer) = Set(answer) == Set(istar(pep, θ))

# closest alternative and value for all pieces
function min_alt(pep::Union{Topm, Topm_State}, θ, Vinv)
    topm = istar(pep, θ)
    return minimum(alternative(Vinv, θ, pep.arms[j], pep.arms[k]) for j in topm, k in setdiff(1:narms(pep), topm))
end

# closest alternative and value for active pieces only
function min_alt_active(pep::Topm_State, θ, Vinv)
    topm = istar(pep, θ)
    topm_active = setdiff(topm, pep.found_topm) # we test only the topm arms that are still active
    min_val = Inf
    min_λ = Nothing
    # for each arm j in the above set, we test only those non-topm arms which are still not labeled as worse than j
    for j in topm_active
        for k in setdiff(1:narms(pep), topm, pep.worse_than[j])
            val, λ = alternative(Vinv, θ, pep.arms[j], pep.arms[k])
            if val < min_val
                min_val = val
                min_λ = λ
            end
        end
    end
    return min_val, min_λ
end

#################################################################
# Online sign identification (OSI)
#################################################################

struct OSI
    arms;  # array of size K of arms in R^d
end

struct OSI_State
    arms;  # array of size K of arms in R^d
    active_arms::Set;  # indexes of those arms whose sign has still to be learned
    found_signs::Array;
    elim_times::Array;
    function OSI_State(arms)
        new(arms, Set(1:length(arms)), zeros(length(arms)), zeros(length(arms)))
    end
end

# factory for creating pep states
init_pep_state(pep::OSI) = OSI_State(pep.arms)

# reset active arms
function reset_pep_state(pep::OSI_State)
    union!(pep.active_arms, Set(1:length(arms)))
end

nanswers(pep::Union{OSI, OSI_State}) = 2^length(pep.arms);
narms(pep::Union{OSI, OSI_State}) = length(pep.arms);
nactive(pep::OSI_State) = length(pep.active_arms);
istar(pep::Union{OSI, OSI_State}, θ) = [arm'θ >= 0 ? 1 : 0 for arm in pep.arms]
is_correct(pep::Union{OSI, OSI_State}, θ, answer) = answer == istar(pep, θ)

# closest alternative and value for a subset of arm indexes (used for min_alt and min_alt_active)
function min_alt_subset(pep::Union{OSI, OSI_State}, θ, Vinv, arm_subset)
    signs = istar(pep,θ)
    min_val = Inf
    min_λ = Nothing
    for j in arm_subset
        if signs[j] == 1
            arm1 = pep.arms[j]
            arm2 = zeros(length(arm1))
        else
            arm2 = pep.arms[j]
            arm1 = zeros(length(arm2))
        end
        val, λ = alternative(Vinv, θ, arm1, arm2)
        if val < min_val
            min_val = val
            min_λ = λ
        end
    end
    return min_val, min_λ
end

# closest alternative and value for all pieces
function min_alt(pep::Union{OSI, OSI_State}, θ, Vinv)
    return min_alt_subset(pep, θ, Vinv, 1:narms(pep))
end

# closest alternative and value for active pieces only
function min_alt_active(pep::OSI_State, θ, Vinv)
    return min_alt_subset(pep, θ, Vinv, pep.active_arms)
end