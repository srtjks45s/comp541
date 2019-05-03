include("./includes.jl")

function getind(word; max=400000, root=false)
    abc = get(wembed,lowercase(word),-1)
    if root == true
        return max-1
    elseif (abc >= 0)
        return abc
    else
        return max
    end
end

lents(datdat) = map(x->length(x[1]), datdat)

function minib(data4c,batchsize)
    datdat = sort(data4c, by=x->length(x[1]), rev=true)
    lentsdatdat = lents(datdat)
    uniqdat = unique(lentsdatdat)
    lennnum = [(i,count(x->x==i,lentsdatdat),j) for (j,i) in enumerate(uniqdat)]
    i=1
    j=0
    k=0
    batdat = []
    for (a,b,c) in lennnum
        bb = b
        j += b
        j = i+b
        k=j
        while bb >= batchsize
            j =i+batchsize-1
            
            push!(batdat, (cat([x[1] for x in datdat[i:j]]...,dims=1), cat([x[2] for x in datdat[i:j]]...,dims=2)))
            i=j+1
            bb -= batchsize
        end
        i += bb
    end
    batdat
end

function batchmult_old(w,x;dim=3)
    xb = ndims(x)>=3
    wb = ndims(w)>=3
    if(wb && xb)
        return cat(collect([w[:,:,i]*x[:,:,i] for i in 1:size(x,dim)])..., dims=dim)
    elseif(wb)
        return cat(collect([w[:,:,i]*x for i in 1:size(x,dim)])..., dims=dim)
    elseif(xb)
        return cat(collect([w*x[:,:,i] for i in 1:size(x,dim)])..., dims=dim)
    else
        return wb*xb
    end
end

function batchmult(w,x;dim=3)
    xb = ndims(x)>=3
    wb = ndims(w)>=3
    if(wb && xb)
        return reshape(vcat(collect([w[:,:,i]*x[:,:,i] for i in 1:size(x,dim)])...), size(w,1), size(x,2), :)
    elseif(wb)
        return reshape(vcat(collect([w[:,:,i]*x for i in 1:size(x,dim)])...), size(w,1), size(x,2), :)
    elseif(xb)
        return reshape(vcat(collect([w*x[:,:,i] for i in 1:size(x,dim)])...), size(w,1), size(x,2), :)
    else
        return wb*xb
    end
end

function batchmult_wew(w,x)
    return reshape(hcat(collect([w[:,i,:]*x[:,i,:] for i in 1:size(x,2)])...), size(w,1), size(x,3), :)
end

# For running experiments
function trainresults(file,model; o...)
    if (print("Train from scratch? "); readline()[1]=='y')
        takeevery(n,itr) = (x for (i,x) in enumerate(itr) if i % n == 1)
        r = ((model(dtrn,MyData1()), model(dtst,MyData1()), zeroone(model,dtrn), zeroone(model,dtst))
             for x in takeevery(length(dtrn), progress(adam(model,repeat(dtrn,30)))))
        r = reshape(collect(Float32,flatten(r)),(4,:))
        Knet.save(file,"results",r)
        Knet.gc() # To save gpu memory
    else
        isfile(file) || download("http://people.csail.mit.edu/deniz/models/tutorial/$file",file)
        r = Knet.load(file,"results")
    end
    println(minimum(r,dims=2))
    return r
end

function perdimk(arr, dims)
    ca = Array(arr)
    pa = permutedims(ca,dims)
    return KnetArray(pa)
end