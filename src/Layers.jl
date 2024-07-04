dense(x::GraphNode, w::GraphNode, b::GraphNode, f::Constant, df::Constant) = BroadcastedOperator(dense, x, w, b, f, df)
forward(::BroadcastedOperator{typeof(dense)}, x, w, b, f, df) = f(w * x .+ b)
backward(::BroadcastedOperator{typeof(dense)}, x, w, b, f, df, g) = let
    g = df(w * x .+ b) .* g
    tuple(w' * g, g * x', sum(g, dims=2))
end

vanilla_rnn(x::GraphNode, w::GraphNode, b::GraphNode, hw::GraphNode, states::GraphNode, f::Constant, df::Constant) = BroadcastedOperator(vanilla_rnn, x, w, b, hw, states, f, df)
forward(o::BroadcastedOperator{typeof(vanilla_rnn)}, x, w, b, hw, states, f, df) = let
    if isnothing(states)
        state = zeros(Float32, size(w, 1), size(x, 2))
        o.inputs[5].output = Matrix{Float32}[]
    else
        state = last(states)
    end
    h = f.(w * x .+ hw * state .+ b)
    push!(o.inputs[5].output, h)
    h
end
backward(::BroadcastedOperator{typeof(vanilla_rnn)}, x, w, b, u, states, f, df, g) = let
    dhw = zeros(Float32, size(u))
    db = zeros(Float32, size(b))
    previoust_state = zeros(Float32, size(states[1]))
    dw = zeros(Float32, size(w))
    for state in reverse(states)
        zL = w * x .+ u * state .+ b
        dp = state .+ u * previoust_state
        dt = df(zL) .* dp .* g
        dw .+= dt * x'
        dhw .+= dt * state'
        db .+= mean(dt, dims=2)
        previoust_state = state
    end
    tuple(w' * g, dw, db, dhw, nothing)
end

function update_weights!(graph::Vector, optimizer::Optimizer)
    for node in graph
        if isa(node, AutomaticDifferention.Variable)
            if !isnothing(node.gradient)
                node.output .-= optimizer(node.gradient)
                node.gradient .= 0
            elseif !isnothing(node.gradient)
                node.output = nothing
            end
        end
    end
end