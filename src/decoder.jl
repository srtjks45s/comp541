include("./includes.jl")
include("./utils1.jl")

struct Biaff; mlphead; mlpdep; w; b; end;
#Biaff(mlph::MLP, mlpd::MLP, input::Int, output::Int)=Biaff(mlph, mlpd , param(output,input), param0(output))
Biaff(rhid::Int, mlph::Int, m::Int, input::Int, output::Int) =
    Biaff(MLP(rhid,mlph, m) , MLP(rhid,mlph, m) , param(m,m), param0(m))
Biaff(rhid::Int, mlph::Int, m::Int) =
    Biaff(MLP(rhid,mlph, m) , MLP(rhid,mlph, m) , param(m,m), param0(m))

(bi::Biaff)(x) = begin
    tts = size(x,3)
    bbs = size(x,2)
    Knet.@timeit timo "mlphead" archead = bi.mlphead(x)
    Knet.@timeit timo "mlpdep" arcdep = bi.mlpdep(x[:,:,2:end])
    hms = size(archead,1)
    Knet.@timeit timo "perdim1" archeadp = perdimk(reshape(archead, hms, bbs, tts), [3, 1, 2])
    Knet.@timeit timo "perdim2" archeadpp = perdimk(reshape(archead, hms, bbs, tts), [3, 2, 1])
    archeadppr = reshape(archeadpp, bbs*tts, hms)
    Knet.@timeit timo "WH" WH = bi.w * arcdep
    Knet.@timeit timo "perdim3" WHp = perdimk(reshape(WH, size(WH,1), bbs, tts-1), [1,3,2])
    Knet.@timeit timo "HWH(bmm)" HWH = Knet.bmm(archeadp, WHp)
    Hb = archeadppr * bi.b 
    Hbr = reshape(Hb, tts, 1 , bbs)
    S = HWH .+ Hbr
end