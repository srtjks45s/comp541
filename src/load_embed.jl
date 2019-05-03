include("./includes.jl")
include("./utils1.jl")

function load_embed(path)
    wembed, wembedind = open(path) do f
        wembed = Dict()
        wembedind = []
        for i in enumerate(eachline(f))
            line = i[2]
            tokens = split(line)
            key = tokens[1]
            temp = Array{Float32, 1}()
            for token in tokens[2:end]
                tmp = tryparse(Float32, token)
                append!(temp, tmp)
            end
            wembed[key] = i[1]
            push!(wembedind,temp)
        end
        wembed, wembedind
    end
    wembed, wembedind
end