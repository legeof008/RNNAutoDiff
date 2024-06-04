include("AutomaticDifferention.jl")
using AutomaticDifferention.GraphDifferention

Wh  = Variable(randn(10,2), name="wh")
Wo  = Variable(randn(1,10), name="wo")
x = Variable([1.98, 4.434], name="x")
y = Variable([0.064], name="y")
losses = Float64[]

function dense(u,x,v,h,b)
    return u.* x + v.*h +b
end