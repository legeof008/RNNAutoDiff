include("GraphDifferention.jl")
module AutomaticDifferention

    using ..GraphDifferention
    using ExportAll
    using Distributions
    using LinearAlgebra:I


    abstract type NetworkLayer end

    struct DenseLayerSoftmax <: NetworkLayer
        ordered_computation_graph :: Vector{GraphNode}
        w_handle :: GraphNode
        x_handle :: GraphNode
        output_handle :: GraphNode
        prediction_handle :: GraphNode
        DenseLayerSoftmax(input_output_pair,test_data) = let 
            
            input_num = input_output_pair.first
            output_num = input_output_pair.second

            @assert length(test_data) == output_num

            ordered_graph, w_handle, x_handle, output_handle, prediction_handle = SoftmaxConnectedLayer(input_num,output_num,test_data)
            new(ordered_graph,w_handle,x_handle,output_handle, prediction_handle)
        end
    end

    struct RnnVanillaTanh <: NetworkLayer
        ordered_computation_graph :: Vector{GraphNode}
        w_handle :: GraphNode
        u_handle :: GraphNode
        x_handle :: GraphNode
        h_handle :: GraphNode
        output_handle :: GraphNode
        RnnVanillaTanh(input_output_pair) = let 
            
            input_num = input_output_pair.first
            output_num = input_output_pair.second

            ordered_graph, h_handle, w_handle, u_handle, x_handle, output_handle = RnnVanillaLayer(input_num,output_num)
            new(ordered_graph,w_handle,u_handle,x_handle,h_handle,output_handle)
        end
    end

    function load_data!(layer::NetworkLayer, data)
        layer.x_handle.output = data
    end

    function load_output_as_h!(layer::RnnVanillaTanh)
        if !isnothing(layer.output_handle.output)
            layer.h_handle.output = layer.output_handle.output
        end
    end

    handle_batching_preperations!(layer::DenseLayerSoftmax) = println("Loss = $(layer.output_handle.output)")
    handle_batching_preperations!(layer::RnnVanillaTanh) = load_output_as_h!(layer)

    function RnnVanillaLayer(input_number,outputs_number)
        x = Variable(ones(input_number,1), name = "x-rnn")
        u = Variable(rand(Uniform(-0.01,0.01),outputs_number,input_number), name = "u-rnn")

        h = Variable(rand(Uniform(-0.01,0.01),outputs_number,1), name = "h-rnn")
        w = Variable(rand(Uniform(-0.01,0.01),outputs_number,outputs_number), name = "w-rnn")

        b = Constant(rand(Uniform(-0.01,0.01),outputs_number,1))
        o = (u*x .+ w*h) .+ b

        activation = tanh(o)
        order = topological_sort(activation)

        #forward!(order)
        return order, h, w, u, x, last(order)
    end

    function SoftmaxConnectedLayer(input_number,outputs_number,test_data)
        b = Constant(rand(Uniform(-0.01,0.01),outputs_number,1))
        x = Variable(ones(input_number,1), name = "x-dense")

        w = Variable(rand(Uniform(-0.01,0.01),outputs_number,input_number), name = "w-dense")
        test = Constant(test_data)

        o = (w*x) .+ b
        activation = softmax(o)
        loss = crossentropy(activation,test)
        order = topological_sort(loss)

        #forward!(order)
        return order, w, x, last(order), activation
    end

    function forward_net!(input, layers...)
        @assert length(layers) > 1 "This function can be run for at least two layers."
        first_layer = layers[1]
        other_layers = layers[2:end]
        first_layer.x_handle.output = input

        output_from_first_layer = forward!(first_layer.ordered_computation_graph)
        other_layers[1].x_handle.output = output_from_first_layer

        for iter in eachindex(other_layers)
            
            current_layer_output = forward!(other_layers[iter].ordered_computation_graph)

            if iter + 1 < length(other_layers)
                other_layers[iter+1].x_handle.output = current_layer_output
            end
        end

        return last(other_layers).output_handle.output
    end

    function forward_net!(input, layer)
        layer.x_handle.output = input

        output_from_layer = forward!(layer.ordered_computation_graph)

        return output_from_layer
    end

    function backward_net!(layers...)
        @assert length(layers) > 1 "This function can be run for at least two layers."
        reversed_layers = reverse(layers)
        last_layer = reversed_layers[1]
        other_layers = reversed_layers[2:end]

        backward!(last_layer.ordered_computation_graph)
        gradient_from_last_layer = last_layer.x_handle.gradient

        for iter in eachindex(other_layers)

            backward!(other_layers[iter].ordered_computation_graph,gradient_from_last_layer)
            current_layer_gradient = other_layers[iter].x_handle.gradient

            if iter + 1 < length(other_layers)
                backward!(other_layers[iter+1],current_layer_gradient)
                gradient_from_last_layer = other_layers[iter+1].x_handle.gradient
            end
        end

        return last(other_layers).output_handle.gradient
    end

    function backward_net!(layer)
        backward!(layer.ordered_computation_graph)
        gradient_from_last_layer = layer.x_handle.gradient

        return gradient_from_last_layer
    end

    function load_batch_of_data!(input_batch,network)
        for layer in network
            handle_batching_preperations!(layer)
        end
        forward_net!(input_batch,network...)
    end
    
    function run_through_batched_data!(batched_data,network)
        for data_batch in batched_data
            load_batch_of_data!(data_batch, network)
        end
        backward_net!(network...)
        println("Predictions vector:")
        last(network).prediction_handle.output
    end


    function learning_step_first_run!(xᵢ, ∇fxᵢ, H⁻¹ᵢ, α = 0.01)
        error("Not usable with matrices")
        # Run the graph forward and backward for the first time just to get new step
        p = -H⁻¹ * ∇fxᵢ;
        xᵢ₊₁ = xᵢ + α*p # this will be the new weight matrix
        return xᵢ₊₁
    end

    function learning_step_second_run!(xᵢ, xᵢ₊₁, ∇fxᵢ, ∇fxᵢ₊₁, H⁻¹ᵢ, α = 0.01)
        error("Not usable with matrices")
        # Run the graph forward and backward again for the second time to approximate Hessian
        yᵢ = ∇fxᵢ - ∇fxᵢ₊₁
        sᵢ = xᵢ₊₁ - xᵢ
        a₁ = (sᵢ * yᵢ')/(yᵢ' * sᵢ)
        a₂ = (yᵢ * sᵢ')/(sᵢ' * yᵢ)
        a₃ = (sᵢ * sᵢ')/(yᵢ' * sᵢ)
        H⁻¹ᵢ₊₁ = (I - a₁) * H⁻¹ᵢ * (I - a₂) + a₃
        return H⁻¹ᵢ₊₁
    end

    @exportAll
end # module AutomaticDifferention

