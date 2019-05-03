include("./includes.jl")
include("./utils1.jl")
include("./decoder.jl")


fModel(tembed::Int, rnnh::Int, mlp1h::Int, mlp1o::Int; o...) = 
    Chain(RNN(tembed, rnnh; bidirectional=true, rnnType = :lstm,o...), Biaff(2*rnnh,mlp1h,mlp1o))
	
fModel2(wmat, tembed::Int, rnnh::Int, mlp1h::Int, mlp1o::Int; o...) = 
    Chain(Embed(wmat, EmbMatT()), RNN(tembed, rnnh; bidirectional=true, rnnType = :lstm,o...), Biaff(2*rnnh,mlp1h,mlp1o))
	
fModel3(wmat, tembed::Int, rnnh::Int, mlp1h::Int, mlp1o::Int; o...) = 
    Chain(Embed(wmat, KnetArray, EmbMatPar()), RNN(tembed, rnnh; bidirectional=true, rnnType = :lstm,dropout=0.3, o...), Biaff(2*rnnh,mlp1h,mlp1o))
	
