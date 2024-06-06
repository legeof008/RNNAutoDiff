using AutomaticDifferention
using Test, MLDatasets, Flux

@testset "ConstructorsTests" begin
    @test begin
        # given - when
        rnn = RnnVanillaTanh(196 => 64)

        # then
        typeof(rnn) == RnnVanillaTanh

        !isnothing(rnn.w_handle)
        size(rnn.w_handle.output) == (64,196)

        !isnothing(rnn.u_handle)
        size(rnn.u_handle.output) == (64,196)

        !isnothing(rnn.x_handle)
        size(rnn.x_handle.output) == (196,1)

        !isnothing(rnn.h_handle)
        size(rnn.h_handle.output) == (196,1)

        isnothing(rnn.output_handle.output)
        
        typeof(rnn.output_handle) == AutomaticDifferention.GraphDifferention.
            ScalarOperator{typeof(tanh)}
    end
    @test begin
        # given - when
        dense = DenseLayerSoftmax(64 => 10, ones(10,1))

        # then
        typeof(dense) == DenseLayerSoftmax

        !isnothing(dense.w_handle)
        size(dense.w_handle.output) == (10,64)

        !isnothing(dense.x_handle)
        size(dense.x_handle.output) == (64,1)

        !isnothing(dense.prediction_handle)

        isnothing(dense.output_handle.output)

        typeof(dense.output_handle) == AutomaticDifferention.GraphDifferention.BroadcastedOperator{typeof(AutomaticDifferention.GraphDifferention.crossentropy)}
    end
    @test begin
        # given
        train_data = MLDatasets.MNIST(split=:train)
        test_data  = MLDatasets.MNIST(split=:test)
            # Prepare data
        x1dim = reshape(train_data.features, 28 * 28, :)
        yhot  = Flux.onehotbatch(train_data.targets, 0:9)
        in1 = x1dim[:,1][197:392]
        yhot1 = yhot[:,1]

        # when
            # Prepare network
        rnn = RnnVanillaTanh(196 => 64)
        dense = DenseLayerSoftmax(64 => 10, yhot1)

        loss = forward_net!(in1,rnn,dense)
        loss_gradient = backward_net!(rnn,dense)
        prediction = dense.prediction_handle.output
        
        # then
        maximum(loss) < 1.
        minimum(loss) > -1.
        
        maximum(loss_gradient) < 1.
        minimum(loss_gradient) > -1.

        size(prediction) == (10,1)
    end

end