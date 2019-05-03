using Pkg; haskey(Pkg.installed(),"Knet") || Pkg.add("Knet")
using Statistics: mean
using Base.Iterators: cycle
using Knet: Knet, AutoGrad, Data, param, param0, mat, RNN, dropout, value, nll, adam, minibatch, progress!, converge, Random

using Knet: TimerOutputs

# Set display width, load packages, import symbols
ENV["COLUMNS"]=72
using Pkg; for p in ("Knet","Plots"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using Knet: Knet, dir, zeroone, progress, sgd, load, save, gc, Param, KnetArray, gpu, Data, nll, relu, training, dropout # param, param0, xavier
using Statistics: mean
using Base.Iterators: flatten

import LinearAlgebra
using Plots; default(fmt=:png,ls=:auto)

