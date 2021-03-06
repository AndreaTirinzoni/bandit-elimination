struct FTL
    L :: Array{Float64, 1};

    function FTL(K)
        new(zeros(K));
    end
end

function act(h::FTL)
    u = h.L .== minimum(h.L);
    u ./ sum(u);
end

function incur!(h::FTL, l)
    h.L .+= l;
end



mutable struct AdaHedge
    L :: Array{Float64, 1};
    Δ :: Float64;

    function AdaHedge(K)
        new(zeros(K), 0.01);
    end
end

function act(h::AdaHedge)
    η = log(length(h.L))/h.Δ;
    u = exp.(-η.*(h.L .- minimum(h.L)));
    u ./ sum(u);
end

function incur!(h::AdaHedge, l)
    u = act(h);
    η = log(length(h.L))/h.Δ;
    h.L .+= l;
    m = minimum(l) - 1/η*log(u'exp.(-η.*(l .- minimum(l))));
    # println(η, " ", u, " ", u'l, " ", -1/η*log(u'exp.(-η.*l)));
    @assert isfinite(m) && u'l ≥ m - 1e-7 "hedge loss $(u'l), Δ=$(h.Δ), η=$η, mix loss $m, u=$u, L=$(h.L)";
    h.Δ  += u'l - m;
end











# Menard calls this "Lazy Mirror Descent" As perhaps the main feature
# here is the mixing step, we call it "Fixed Share".  This is the
# version with slowly decreasing O(1/sqrt(t)) learnign and switching
# rates.  The switching is used to force exploration
mutable struct FixedShare
    w :: Array{Float64, 1};
    t; # time
    S; # loss range: S ≥ \max_k l_k - \min_k l_k
       # TODO: adapt to it (like AdaHedge?)

    function FixedShare(K; S=1)
        new(fill(1/K,K), 0, S);
    end
end

function act(h::FixedShare)
    h.w
end

function incur!(h::FixedShare, l)
    K = length(l);
    h.t += 1;

    # Exponential Weights update
    η = sqrt(log(K)/h.t)/h.S;   # decaying learning rate
    h.w .*= exp.(-η*(l .- minimum(l)));
    h.w ./= sum(h.w);

    # Fixed Share (to unifor prior)
    γ = 1/(4*sqrt(h.t)); # decaying switching rate
    h.w .= (1-γ).*h.w .+ γ/K;
end
