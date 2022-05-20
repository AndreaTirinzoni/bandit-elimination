function sherman_morrison(Vinv, u, v)
    num = (Vinv*u)*transpose(transpose(Vinv)*v)
    denum = 1 + transpose(v)*Vinv*u
    return Vinv .- num / denum
end

function sherman_morrison(Vinv, u)
    Vinv_u = Vinv*u
    num = Vinv_u*transpose(Vinv_u)
    denum = 1 + transpose(u)*Vinv_u
    return Vinv .- num / denum
end

function alternative(Vinv, θ, arm1, arm2)
    # computes closest alternative and value in halfspace {λ : arm2'λ >= arm1'λ}
    # for bai and top-m, the arms should be s.t. arm1'θ > arm2'θ (ie, arm1 is better than arm2)
    # for thresholding, if the target arm is ϕ: (TODO: check)
    #    - set (arm1 = 0, arm2 = ϕ) if sign(ϕ'θ) < 0
    #    - set (arm1 = ϕ, arm2 = 0) if sign(ϕ'θ) > 0
    direction = arm1 .- arm2
    gap = (direction'θ)
    if gap <= 0
        return 0, θ
    end
    norm = ((direction')*Vinv*direction)
    λ = θ .- (gap / norm) * Vinv * direction
    val = .5 * gap^2 / norm
    return val, λ
end

function check_power2(t)
    exponent = floor(Int, log2(t));
    return (t == 2^exponent);
end