# Some thresholds, implemented as types so we can save them with JLD2
# (which does not allow saving functions directly)


# "Heuristic" threshold from Optimal Best Arm Identification with Fixed Confidence
# (Garivier and Kaufmann 2016). Recommended in section 6.
struct GK16
end

function (β::GK16)(t, δ)
    log((log(t)+1)/δ)
end

# Slightly larger heuristic threshold that GK16
struct HeuristicThreshold
end

function (β::HeuristicThreshold)(t, δ)
    log((t+1)/δ)
end

# "Theoretical" threshold for linear bandits from "Improved Algorithms for Linear Stochastic Bandits"
# (Abbasi-Yadkori et al., 2011, Theorem 2, second statement).
struct LinearThreshold
    d::Int64 # feature dimension
    L::Float64 # max feature norm
    S::Float64 # max param norm
    λ::Float64 # regularization
end

function (β::LinearThreshold)(t, δ)
    log_term = (β.d / 2) * log(1 + t * β.L^2 / (β.λ * β.d))
    return (sqrt(log(1/δ) + log_term) + sqrt(β.λ / 2) * β.S) ^ 2
end

#######
# TEST
#######
# δ = 0.1
# β = GK16()
# for t = 1:10
#     println(t, " ", β(t,δ))
# end

# println()

# β = LinearThreshold(2, 1, 1, 1)
# for t = 1:10
#     println(t, " ", β(t,δ))
# end