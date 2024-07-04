# Automatic Differentiation
This libary is a utility aimed at performing Vanilla RNN ML. It consists of a backward propagation of gradient implementation for Vanilla RNNS.
## Example usage 
This is a barebones implementation for which data loading can be tuned at will. For example :
```julia
function load_data(batch_size)
    printstyled("Loading data...\n", color = :yellow)
    xt, yt = DataTransformationUtils.prepare_and_encode(:train; one_hot = true)
    train_x_batched = DataTransformationUtils.batch_data(xt, batch_size)
    train_y_batched = DataTransformationUtils.batch_data(yt, batch_size)
    xv, yv = DataTransformationUtils.prepare_and_encode(:test; one_hot = true)
    return xt, yt, train_x_batched, train_y_batched, xv, yv
end
```
Then the learing process could look something like this:
 ```
 function train_network(epochs,batch_size)
    xt, yt, xt_batched, yt_batched, xv, yv = load_data(batch_size)

    # Input

    x = AutomaticDifferention.Variable([0.], name="x")

    # Dense Layer

    wd = AutomaticDifferention.Variable(glorot_uniform(10, 64))
    bd = AutomaticDifferention.Variable(glorot_uniform(10, ))
    fd = AutomaticDifferention.Constant(x -> x)
    dfd = AutomaticDifferention.Constant(x -> ones(size(x)))

    # RNN Layer

    wr = AutomaticDifferention.Variable(glorot_uniform(64, 196))
    br = AutomaticDifferention.Variable(glorot_uniform(64, ))
    u = AutomaticDifferention.Variable(glorot_uniform(64, 64))
    states = AutomaticDifferention.Variable(nothing, name = "hstates")
    fr = AutomaticDifferention.Constant(tanh)
    dfr = AutomaticDifferention.Constant(AutomaticDifferention.tanh_derivative)

    optimizer = AutomaticDifferention.Descent(10e-3)

    vanilla_rnn = AutomaticDifferention.vanilla_rnn(x, wr, br, u, states, fr, dfr)
    dense = AutomaticDifferention.dense(vanilla_rnn, wd, bd, fd, dfd)
    ordered_computation_graph = AutomaticDifferention.topological_sort(dense)
    
    batch_loss = Float64[]
    batch_acc = Float64[]

    printstyled("Training...\n", color = :yellow)
    for epoch in 1:epochs
        batches = randperm(size(xt_batched, 1))
        @time for batch in batches
            states.output = nothing
            x.output = xt_batched[batch][1:196,:]
            AutomaticDifferention.forward!(ordered_computation_graph)

            x.output = xt_batched[batch][197:392,:]
            AutomaticDifferention.forward!(ordered_computation_graph)

            x.output = xt_batched[batch][393:588,:]
            AutomaticDifferention.forward!(ordered_computation_graph)

            x.output = xt_batched[batch][589:end,:]
            result = AutomaticDifferention.forward!(ordered_computation_graph)

            loss = AutomaticDifferention.loss(result, yt_batched[batch])
            acc = DataTransformationUtils.accuracy(result, yt_batched[batch])

            push!(batch_loss, loss)
            push!(batch_acc, acc)

            gradient = AutomaticDifferention.softmax_crossentropy_gradient(result, yt_batched[batch]) ./ batch_size
            AutomaticDifferention.backward!(ordered_computation_graph, seed=gradient)
            AutomaticDifferention.update_weights!(ordered_computation_graph, optimizer)
        end
        states.output = nothing
        test_graph = AutomaticDifferention.topological_sort(dense)

        x.output = xv[  1:196,:]
        AutomaticDifferention.forward!(test_graph)

        x.output = xv[197:392,:]
        AutomaticDifferention.forward!(test_graph)

        x.output = xv[393:588,:]
        AutomaticDifferention.forward!(test_graph)

        x.output = xv[589:end,:]
        result = AutomaticDifferention.forward!(test_graph)

        loss = AutomaticDifferention.loss(result, yv)
        acc = DataTransformationUtils.accuracy(result, yv)

        @show epoch loss acc
    end
    return batch_loss, batch_acc
end
 ```