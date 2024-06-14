module DataTransformationUtils
    using Flux:onecold
    using MLDatasets, Random
    using Statistics: mean
    using Random

    Random.seed!(1235)
    
    function accuracy(ŷ, y)
        pred_classes = onecold(ŷ)
        true_classes = onecold(y)
        return round(100 * mean(pred_classes .== true_classes); digits=2)
    end

    function prepare_and_encode(split::Symbol; one_hot::Bool=true)
        features, targets = load(split)
        x1dim = reshape(features, 28 * 28, :)
        yhot = one_hot ? one_hot_encode(targets, 0:9) : targets

        perm = randperm(size(x1dim, 2))
        x1dim = x1dim[:, perm]
        yhot = yhot[:, perm]
        return x1dim, yhot
    end

    function load(split::Symbol)
        data = MLDatasets.MNIST(split = split)
        return data.features, data.targets
    end

    function one_hot_encode(targets, classes)
        one_hot = zeros(Int, length(classes), length(targets))
        for (i, class) in enumerate(classes)
            one_hot[i, findall(x -> x == class, targets)] .= 1
        end
        return one_hot
    end

    function batch_data(data, n)
        arr = Array{Matrix{Float32}}(undef, 0)
        for i in 1:size(data,2) / n
            i = floor(Int, i)
            push!(arr, data[:, (i * n - n + 1):i*n])
        end
        return arr
    end
end