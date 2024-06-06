using AutomaticDifferention.GraphDifferention
import LinearAlgebra: mul!
import Base:sum
using Test

@testset "ConstructorsTests" begin
    @test begin
        # given
        x = Variable(5.)
        # then
        x.output == 5.
        isnothing(x.gradient)
    end
    @test begin
        # given
        x = Variable(5.)
        y = sin(x)
        # then
        typeof(y) == ScalarOperator{typeof(sin)}
    end
    @test begin
        # given
        x = Variable(5.)
        two = Constant(2.)
        y = x^two
        # then
        typeof(y) == ScalarOperator{typeof(^)}
    end
    @test begin
        # given
        x = Variable(5.)
        y = x*x
        # then
        typeof(y) == BroadcastedOperator{typeof(mul!)}
    end
    @test begin
        # given
        x = Variable(5.)
        y = x.*x
        # then
        typeof(y) == BroadcastedOperator{typeof(*)}
    end
    @test begin
        # given
        x = Variable(5.)
        y = sum(x)
        # then
        typeof(y) == BroadcastedOperator{typeof(sum)}
    end
    @test begin
        # given
        x = Variable(5.)
        y = x./x
        # then
        typeof(y) == BroadcastedOperator{typeof(/)}
    end
    @test begin
        # given
        x = Variable(5.)
        y = max.(x,x)
        # then
        typeof(y) == BroadcastedOperator{typeof(max)}
    end
end

@testset "ForwardOperationsTest" begin
    @test begin
        # given
        x = Constant(5.)
        # then
        isnothing(reset!(x))
    end
    @test begin
        # given
        c = Constant(5.)
        x = Variable(5.)
        y = sin(x)
        x.gradient = 1.
        y.gradient = 1.
        # when
        reset!(x)
        reset!(y)
        # then
        isnothing(reset!(c))
        isnothing(x.gradient)
        isnothing(y.gradient)
    end
    @test begin
        # given
        c = Constant(5.)
        x = Variable(5.)    
        # then
        isnothing(compute!(c))
        isnothing(compute!(x))
    end
    @test begin
        # given
        x = Variable(5.)
        y1 = sin(x)
        y2 = x^2.
        y3 = x*x
        y4 = x.*x
        y5 = sum(x)
        y6 = x./x
        y7 = max.(x,x)

        # when
        compute!(y1)
        compute!(y2)
        compute!(y3)
        compute!(y4)
        compute!(y5)
        compute!(y6)
        compute!(y7)

        # then
        y1.output == 
            forward(y1, x.output)
        y2.output == 
            forward(y2, x.output, 2.)
        y3.output == 
            forward(y3, x.output, x.output)     
        y4.output == 
            forward(y4, x.output, x.output)
        y5.output == 
            forward(y5, x.output)
        y6.output == 
            forward(y6, x.output, x.output)
        y7.output == 
            forward(y7, x.output, x.output)
    end
end

@testset "ForwardSortingAndComputationTest" begin
    @test begin
        # given
        x = Variable(5.)
        y = x*x
        z = sin(y)
        # when
        result = topological_sort(z)
        # then
        first(result) == x
        last(result) == z
        y ∈ result
    end
    @test begin
        # given
        x = Variable(5.)
        y = x*x
        z = sin(y)
        # when
        order = topological_sort(z)
        result = forward!(order)
        # then
        y.output == 25.
        z.output = sin(25.)
        result == sin(25.)
    end
end

@testset "BackwardsComputationTest" begin
    @test begin
        # given
        c = Constant(5.)
        x = Variable(5.)    
        y = Variable(5.)
        gradient = 1.
        x.gradient = nothing
        y.gradient = [1.,1.]
        # when
        update!(x,gradient)
        update!(y,gradient)
        # then
        isnothing(update!(c,gradient))
        x.gradient == gradient
        y.gradient == [2.,2.]
    end
    @test begin
        # given
        x = Variable(5.)
        y1 = sin(x)

        # when
        forward!(topological_sort(y1))

        y1.gradient = 1.

        backward!(y1)

        # then
        x.gradient[1] ==  
            backward(y1,[input.output for input in y1.inputs]..., y1.gradient)[1]

    end
    @test begin
        # given
        x = Variable(5.)
        y1 = sin(x)

        # when
        order = topological_sort(y1)
        forward!(order)
        backward!(order)

        # then
        first(order).gradient[1] == cos(5.) == x.gradient[1]
        last(order).gradient[1] == 1.

    end
    @test begin
        # given
        x = Variable(5.)
        y1 = x ^ Constant(2.)
        y2 = sin(y1)

        # when
        order = topological_sort(y2)
        forward!(order)
        backward!(order)

        # then
        first(order).gradient[1] == 10. * cos(25.) == x.gradient[1]
        last(order).gradient[1] == 1.
        y1.gradient[1] == cos(25.)
    end
end
@testset "ActivationLayerTest" begin
    @test begin
        # given 
        x = Variable([5.;5.], name = "x")
        u = Variable(rand(2,2), name = "u")
        b = Variable([5.;5.], name = "b")
        o = u*x .+b
        activation = tanh(o)
        # when
        order = topological_sort(activation)
        forward!(order)
        backward!(order)
        # then 
        dLdY = order[5].gradient
        dLdU = dLdY * x.output'
        dLdX = u.output' * dLdY
        dLdB = dLdY

        dLdY == ones(2) .- tanh.(order[5].output).^2
        dLdU == u.gradient
        dLdX == x.gradient
        dLdB == b.gradient
    end
    @test begin
        # given 
        x = Variable([20.;5.], name = "x")
        activation = softmax(x)
        # when
        order = topological_sort(activation)
        forward!(order)
        backward!(order)
        softmax_op = last(order)

        # then 
        softmax_op.output == softmax(x.output)
        sum(softmax_op.output) ≈ 1

        dLdX1 = -softmax(x.output)[1] * (sum(softmax_op.output .* softmax_op.gradient) - softmax_op.gradient[1])
        dLdX1 == x.gradient[1]

    end
end

@testset "LossFunctionTest" begin
    @test begin
        # given
        x = Variable([20.;5.], name = "x")
        target = Constant([0.,1.])
        loss = crossentropy(x,target)

        # when
        order = topological_sort(loss)
        forward!(order)
        backward!(order)

        loss_op = last(order)

        # then
        loss_op.output == crossentropy(x.output,target.output)

        𝟏 = ones(length(x.output))
        x.gradient == -(target.output./x.output) + (𝟏.-target.output)./(𝟏.-x.output)
    end
end