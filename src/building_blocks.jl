include("./includes.jl")
include("./utils1.jl")


struct MyData1 end
struct EmbMatT end
struct EmbMatPar end


struct Embed; w; end

Embed(vocab::Int,embed::Int)=Embed(param(embed,vocab))
Embed(wmat, ::EmbMatT) = Embed(wmat)
Embed(wmat, aType, ::EmbMatPar) = Embed(param(wembedmat, atype = aType))
Embed(wmat, ::EmbMatPar) = Embed(param(wembedmat))

(e::Embed)(x) = e.w[:,x]  # (B,T)->(X,B,T)->rnn->(H,B,T)


struct Dense; w; b; f; end
Dense(i::Int,o::Int,f=identity) = Dense(param(o,i), param0(o), f)
(d::Dense)(x) = d.f.(d.w * mat(x,dims=1) .+ d.b)

struct Linear; w; b; end

Linear(input::Int, output::Int)=Linear(param(output,input), param0(output))

(l::Linear)(x) = l.w * mat(x,dims=1) .+ l.b  # (H,B,T)->(H,B*T)->(V,B*T)
(l::Linear)(x,y)= quadl(l(x),y)[1]



struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)
(c::Chain)(d::Data) = mean(c(x,y) for (x,y) in d)
(c::Chain)(d,a,b) = mean(c(x,y) for (x,y) in d)
(c::Chain)(d, ::MyData1) = mean(c(x,y) for (x,y) in d)


MLP(i::Int, h::Int, o::Int) = Chain(Dense(i,h, Knet.relu), Dense(h,o,Knet.relu))

