export Optimizer, Descent
    
abstract type Optimizer end

struct Descent <: Optimizer
    learning_rate
end