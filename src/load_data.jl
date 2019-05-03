function load_data3(path, UDEPREL)
    xtrain, ytrain, deprels = open(path) do f
    xtrain = []
    ytrain = []
    deprels = []
    sentence = []
    temprels = []
    arcs = []
    count = 1
    for i in enumerate(eachline(f))  
      if i[2] == ""
        push!(xtrain, sentence)
        labels = zeros(count, count)
        push!(ytrain, arcs)
        push!(deprels, temprels)
      elseif i[2][1] != '#'
        temp = split(i[2])
        if temp[1] == "1"
            sentence = []
            arcs = []
            temprels = []
            push!(sentence, temp[2])
            push!(arcs, parse(Int64, temp[7]))
            tmp = split(temp[8], ":")
            push!(temprels, UDEPREL[lowercase(tmp[1])])
            count = 1
        else
            if isnumeric(temp[7][1])
                push!(sentence, temp[2]) 
                tmp = split(temp[8], ":")
                push!(temprels, UDEPREL[lowercase(tmp[1])])
                push!(arcs, parse(Int64, temp[7]))
            end
            count += 1
        end
      end
    end
    xtrain, ytrain, deprels
    end
    xtrain, ytrain, deprels
end