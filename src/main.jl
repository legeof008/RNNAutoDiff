include("AutomaticDifferention.jl")
using .AutomaticDifferention
using MLDatasets, Flux
using Statistics:mean

train_data = MLDatasets.MNIST(split=:train)
test_data  = MLDatasets.MNIST(split=:test)

# Prepare data
x1dim = reshape(train_data.features, 28 * 28, :)
yhot  = Flux.onehotbatch(train_data.targets, 0:9)

in1 = x1dim[:,1][1:196]
in2 = x1dim[:,1][197:392]
in3 = x1dim[:,1][393:588]
in4 = x1dim[:,1][589:end]

in12 = x1dim[:,2][1:196]
in22 = x1dim[:,2][197:392]
in32 = x1dim[:,2][393:588]
in42 = x1dim[:,2][589:end]

in13 = x1dim[:,3][1:196]
in23 = x1dim[:,3][197:392]
in33 = x1dim[:,3][393:588]
in43 = x1dim[:,3][589:end]

in14 = x1dim[:,4][1:196]
in24 = x1dim[:,4][197:392]
in34 = x1dim[:,4][393:588]
in44 = x1dim[:,4][589:end]

yhot1 = yhot[:,2]

# Prepare network
rnn = RnnVanillaTanh(196 => 64)
dense = DenseLayerSoftmax(64 => 10, yhot1)
network = [rnn, dense]

run_through_batched_data!([in1, in2, in3, in4],network)
run_through_batched_data!([in12, in22, in32, in42],network)
run_through_batched_data!([in13, in23, in33, in43],network)
run_through_batched_data!([in14, in24, in34, in44],network)

#update_net_weights!(network)


# Calculate accuracy
ŷ = dense.prediction_handle.output;
y = yhot1;

@show acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)));