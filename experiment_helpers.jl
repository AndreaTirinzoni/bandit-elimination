function solve(pep, μ, δ, β)
    Tstar, wstar = oracle(pep, μ);
    ⋆ = istar(pep, μ);

    # lower bound
    kl = (1-2δ)*log((1-δ)/δ);
    lbd = Tstar*kl;

    # more practical lower bound with the employed threshold β
    practical = binary_search(t -> t-Tstar*β(t), max(1, lbd), 1e10);

    Tstar, wstar, ⋆, lbd, practical;
end


function dump_stats(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, repeats)
    K = narms(pep)
    rule = repeat("-", 60);

    println("");
    println(rule);
    println("$(typeof(pep)) at δ = $δ");
    println(@sprintf("%-30s", "Arm"),
            join(map(k -> @sprintf("%6s", k), 1:K)), " ",
            @sprintf("%7s", "total"), "  ",
            @sprintf("%7s", "err"), "  ",
            @sprintf("%7s", "time"));
    println(@sprintf("%-30s", "a\'θ"),
            join(map(x -> @sprintf("%6.2f", x'θ), pep.arms)));
    println(rule);
    println(rule);

    for r in eachindex(sampling_rules)
        Eτ = sum(x->sum(x[2]), data[r,:])/repeats;
        #err = sum(x->x[1].!=star, data[r,:])/repeats;
        err = sum(x->!is_correct(pep, θ, x[1]), data[r,:])/repeats;
        tim = sum(x->x[3],     data[r,:])/repeats;
        println(@sprintf("%-30s", (long(sampling_rules[r]) * "+" * long(stopping_rules[r]) * long(elim_rules[r]))),
                join(map(k -> @sprintf("%6.0f", sum(x->x[2][k], data[r,:])/repeats), 1:K)), " ",
                @sprintf("%7.0f", Eτ), "  ",
                @sprintf("%7.5f", err), "  ",
                @sprintf("%7.5f", tim/1e9)
                );
        if err > δ
            @warn "too many errors for $(sampling_rules[r])+$(stopping_rules[r])";
        end
    end
    println(rule);
end


function τhist(pep, μ, δ, β, srs, data)
    Tstar, wstar, ⋆, lbd, practical = solve(pep, μ, δ, β)

    stephist(map(x -> sum(x[2]), data)', label=permutedims(collect(abbrev.(srs))));
    vline!([lbd], label="lower bd");
    vline!([practical], label="practical");
end


function boxes(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, variable)

    names = [abbrev(sampling_rules[r]) * "+" * abbrev(stopping_rules[r]) * abbrev(elim_rules[r]) for r in 1:length(stopping_rules)];
    xs = permutedims(collect(names));

    # data has 4-tuples (answer, N, time, elim_times) of size nrepeats on each row, one row per algorithm
    if variable == "samples"
        means = sum(sum.(getindex.(data,2)),dims=2)/repeats;
        points = map(x -> sum(x[2]), data)';
    elseif variable == "time"
        points =  map(x -> sum(x[3])/1e9, data)';
        means = sum(points, dims=1)' ./ repeats;
    elseif variable == "time-iter"
        samples = map(x -> sum(x[2]), data)';
        points =  map(x -> sum(x[3])/1e9, data)' ./ samples;
        means = sum(points, dims=1)' ./ repeats;
    else
        return Nothing;
    end

    plot(legend=:top)

    boxplot!(
        xs,
        points,
        label="",
        notch=true,
        outliers=true)

    plot!(xs, means', marker=(:star4,10,:black), label="");
end

function plot_elim_times(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, step, names=Nothing, max_t=Nothing)

    if names == Nothing
        names = [abbrev(sampling_rules[r]) * "+" * abbrev(stopping_rules[r]) * abbrev(elim_rules[r]) for r in 1:length(stopping_rules)]
    end
    n_algos = length(names)

    # stopping time for each run (n_algos x n_repeats)
    samples = map(x -> sum(x[2]), data);
    if max_t == Nothing
        max_t = maximum(samples)
    end

    x = 1:step:max_t
    n_points = length(x)

    means = zeros(n_algos, n_points)

    for (i, name) in enumerate(names)
        n_active = zeros(n_points, repeats)
        for r in 1:repeats
            elim_times = data[i, r][4]
            elim_times[elim_times .== 0] .= max_t + 1
            sort!(elim_times)
            n = length(elim_times)
            idx = 1
            for j in 1:n_points
                while idx <= length(elim_times) && elim_times[idx] <= x[j]
                    idx += 1
                    n -= 1
                end
                n_active[j, r] = n
            end
        end
        means[i, :] = sum(n_active, dims=2) ./ repeats
    end

    plt = plot(legend=:topright)

    for (i, n) in enumerate(names)
        plot!(x, means[i, :], label=n, linewidth=2)
    end

    return plt
end

function randmax(vector, rank = 1)
   # returns an integer, not a CartesianIndex
    vector = vec(vector)
    Sorted = sort(vector, rev = true)
    m = Sorted[rank]
    Ind = findall(x -> x == m, vector)
    index = Ind[floor(Int, length(Ind) * rand())+1]
    return index
end


function randmin(vector, rank = 1)
   # returns an integer, not a CartesianIndex
    vector = vec(vector)
    Sorted = sort(vector, rev = false)
    m = Sorted[rank]
    Ind = findall(x -> x == m, vector)
    index = Ind[floor(Int, length(Ind) * rand())+1]
    return index
end
