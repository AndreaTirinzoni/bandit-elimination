# A list of stopping rules, including LLR an Elimination stopping
# Each stopping rule is a function returning two values:
#  - a boolean indicating whether the algorithm should stop
#  - the recommended answer if stopping, Nothing otherwise

struct LLR_Stopping end

long(sr::LLR_Stopping) = "LLR";
abbrev(sr::LLR_Stopping) = "L";

struct Elim_Stopping end

long(sr::Elim_Stopping) = "Elim";
abbrev(sr::Elim_Stopping) = "E";

struct NoStopping end

long(sr::NoStopping) = "";
abbrev(sr::NoStopping) = "";

struct Force_Stopping # a wrapper to force stopping when a maximum number of samples is reached
    max_samples::Int64
    base_stopping::Union{LLR_Stopping, Elim_Stopping}
    function Force_Stopping(max_samples, base_stopping) 
        @assert max_samples > 0
        new(max_samples, base_stopping)
    end
end

long(sr::Force_Stopping) = long(sr.base_stopping);
abbrev(sr::Force_Stopping) = abbrev(sr.base_stopping);

# The LLR stopping rule
function stop(criterion::LLR_Stopping, pep, β, t, δ, θ_hat, Vinv)
    val, _ = min_alt(pep,θ_hat, Vinv)
    if val > β(t,δ)
        return true, istar(pep,θ_hat)
    end
    return false, Nothing
end

# Elimination stopping rules
# We need a different function for each pure-exploration problem since the recommendation rules are different

function stop(criterion::Elim_Stopping, pep::BAI_State, β, t, δ, θ_hat, Vinv)
    if length(pep.active_arms) > 1
        return false, Nothing
    end
    answer = length(pep.active_arms) > 0 ? first(pep.active_arms) : rand(1:narms(pep))
    return true, answer
end

function stop(criterion::Elim_Stopping, pep::Topm_State, β, t, δ, θ_hat, Vinv)
    if length(pep.found_topm) < pep.m
        return false, Nothing
    end
    answer = first(pep.found_topm, pep.m)
    return true, answer
end

function stop(criterion::Elim_Stopping, pep::OSI_State, β, t, δ, θ_hat, Vinv)
    if length(pep.active_arms) > 0
        return false, Nothing
    end
    answer = pep.found_signs
    return true, answer
end

# Wrapper to force stopping when max_samples is reached
function stop(criterion::Force_Stopping, pep, β, t, δ, θ_hat, Vinv)
    if t >= criterion.max_samples
        return true, istar(pep,θ_hat)
    end
    return stop(criterion.base_stopping, pep, β, t, δ, θ_hat, Vinv)
end

# Fictitious stopping rule that never stops (used for algorithm with internal stopping, eg those native elimination-based)
function stop(criterion::NoStopping, pep, β, t, δ, θ_hat, Vinv)
    return false, Nothing
end