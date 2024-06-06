module GraphDifferention

    using ExportAll
    using LinearAlgebra:diagm

    abstract type GraphNode end
    abstract type Operator <: GraphNode end

    #=
        Structure types
    =#

    struct Constant{T} <: GraphNode
        output :: T
    end

    mutable struct Variable <: GraphNode
        output :: Any
        gradient :: Any
        name :: String
        Variable(output; name="?") = new(output, nothing, name)
    end

    mutable struct ScalarOperator{F} <: Operator
        inputs :: Any
        output :: Any
        gradient :: Any
        name :: String
        ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
    end

    mutable struct BroadcastedOperator{F} <: Operator
        inputs :: Any
        output :: Any
        gradient :: Any
        name :: String
        BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
    end

    ### Pretty-printing
    ## It helps tracking what happens

    import Base: show, summary

    show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
    show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
    show(io::IO, x::Constant) = print(io, "const ", x.output)
    show(io::IO, x::Variable) = begin
        print(io, "var ", x.name);
        print(io, "\n ┣━ ^ "); summary(io, x.output)
        print(io, "\n ┗━ ∇ ");  summary(io, x.gradient)
    end

    ### Graph building

    function visit(node::GraphNode, visited, order)
        if node ∈ visited
        else
            push!(visited, node)
            push!(order, node)
        end
        return nothing
    end
        
    function visit(node::Operator, visited, order)
        if node ∈ visited
        else
            push!(visited, node)
            for input in node.inputs
                visit(input, visited, order)
            end
            push!(order, node)
        end
        return nothing
    end
    
    function topological_sort(head::GraphNode)
        visited = Set()
        order = Vector()
        visit(head, visited, order)
        return order
    end

    ### Forward pass

    reset!(node::Constant) = nothing
    reset!(node::Variable) = node.gradient = nothing
    reset!(node::Operator) = node.gradient = nothing

    compute!(node::Constant) = nothing
    compute!(node::Variable) = nothing
    compute!(node::Operator) =
        node.output = forward(node, [input.output for input in node.inputs]...)

    function forward!(order::Vector)
        for node in order
            compute!(node)
            reset!(node)
        end
        return last(order).output
    end

    ### Backward pass

    update!(node::Constant, gradient) = nothing
    update!(node::GraphNode, gradient) = if isnothing(node.gradient)
        node.gradient = gradient else node.gradient .+= gradient
    end

    function backward!(order::Vector, seed = ones(length(last(order).output)))
        result = last(order)
        result.gradient = seed
        #@assert length(result.output) == 1 "Gradient is defined only for scalar functions"
        for node in reverse(order)
            backward!(node)
        end
        return nothing
    end

    function backward!(node::Constant) end
    function backward!(node::Variable) end
    function backward!(node::Operator)
        inputs = node.inputs
        gradients = backward(node, [input.output for input in inputs]..., node.gradient)
        for (input, gradient) in zip(inputs, gradients)
            update!(input, gradient)
        end
        return nothing
    end

    # Scalar operators

    import Base: ^
    ^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
    ^(x::GraphNode, n::Number) = ScalarOperator(^, x, Constant(n))
    forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
    backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

    import Base: sin
    sin(x::GraphNode) = ScalarOperator(sin, x)
    forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
    backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))
    
    import Base: tanh
    tanh(x::GraphNode) = ScalarOperator(tanh,x)
    forward(::ScalarOperator{typeof(tanh)}, x) = return tanh.(x)
    backward(node::ScalarOperator{typeof(tanh)}, x, g) = let
        𝟏 = ones(length(node.output))
        x_sqr = node.output.^2
        derivative_tanh_vector = 𝟏 .- x_sqr
        tuple(derivative_tanh_vector .* g)
    end
    # Broadcast operators

    import Base: *
    import LinearAlgebra: mul!
    # x * y (aka matrix multiplication)
    *(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
    forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
    backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

    # x .* y (element-wise multiplication)
    Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
    forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
    backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
        𝟏 = ones(length(node.output))
        Jx = diagm(y .* 𝟏)
        Jy = diagm(x .* 𝟏)
        tuple(Jx' * g, Jy' * g)
    end

    softmax(x) = return exp.(x) ./ sum(exp.(x))
    
    softmax(x::GraphNode) = BroadcastedOperator(softmax,x)
    forward(::BroadcastedOperator{typeof(GraphDifferention.softmax)},x) = return softmax(x)
    backward(node::BroadcastedOperator{typeof(GraphDifferention.softmax)},x,g)  = let
        vector_of_derivatives = Vector()
        for i in 1:length(node.output)
            yi = node.output[i]
            ∑dLdYj_times_Yj = sum(g.* node.output)
            dLdYi = g[i]
            result  =  -yi*(∑dLdYj_times_Yj - dLdYi)
            push!(vector_of_derivatives,result)
        end
        tuple(vector_of_derivatives)
    end

    crossentropy(output,target) =  sum(-target.*log.(output))
    crossentropy(x::GraphNode,y::GraphNode) = BroadcastedOperator(crossentropy,x,y)
    forward(::BroadcastedOperator{typeof(GraphDifferention.crossentropy)},x,y) = return crossentropy(x,y)
    backward(node::BroadcastedOperator{typeof(GraphDifferention.crossentropy)},x,y,g)  = let
        𝟏 = ones(length(x))
        tuple(g.*(-(y./x) + (𝟏.-y)./(𝟏.-x)))
    end


    Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
    forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
    backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g,-g)

    Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
    forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
    backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

    import Base: sum
    sum(x::GraphNode) = BroadcastedOperator(sum, x)
    forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
    backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
        𝟏 = ones(length(x))
        J = 𝟏'
        tuple(J' * g)
    end

    Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
    forward(::BroadcastedOperator{typeof(/)}, x, y) = return x ./ y
    backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) = let
        𝟏 = ones(length(node.output))
        Jx = diagm(𝟏 ./ y)
        Jy = (-x ./ y .^2)
        tuple(Jx' * g, Jy' * g)
    end

    import Base: max
    Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
    forward(::BroadcastedOperator{typeof(max)}, x, y) = return max.(x, y)
    backward(::BroadcastedOperator{typeof(max)}, x, y, g) = let
        Jx = diagm(isless.(y, x))
        Jy = diagm(isless.(x, y))
        tuple(Jx' * g, Jy' * g)
    end

    @exportAll()
end