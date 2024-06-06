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

yhot1 = yhot[:,1]

# Prepare network
rnn = RnnVanillaTanh(196 => 64)
dense = DenseLayerSoftmax(64 => 10, yhot1)
network = [rnn, dense]

run_through_batched_data!([in1, in2, in3, in4],network)

# Calculate accuracy
ŷ = dense.prediction_handle.output;
y = yhot1;

acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)));