using Random
using LinearAlgebra
using ProximalOperators, ShiftedProximalOperators, MLDatasets, RegularizedProblems
using NLPModels, NLPModelsModifiers #ReverseADNLSModels
using RegularizedOptimization
using DataFrames
using SolverBenchmark

include("plot-utils-svm-sto.jl")

# Random.seed!(1234)

function demo_solver(nlp_tr, nls_tr, sampled_nls_tr, sol_tr, nlp_test, nls_test, sampled_nls_test, sol_test, h, χ, suffix="l0-linf"; n_runs::Int = 1, MaxEpochs::Int = 100, MaxTime = 3600.0, version::Int = 4, precision = 1e-4)
    options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = precision, ϵr = precision, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
    suboptions = RegularizedOptimization.ROSolverOptions(maxIter = 100)

    sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, σmax = 1e16, ϵa = precision, ϵr = precision, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
    acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100

    @info "using R2 to solve with" h
    reset!(nlp_tr)
    R2_out = R2(nlp_tr, h, options, x0=nlp_tr.meta.x0)
    nr2 = neval_obj(nlp_tr)
    ngr2 = neval_grad(nlp_tr)
    r2train = residual(nls_tr, R2_out.solution) #||e - tanh(b * <A, x>)||^2, b ∈ {-1,1}^n
    r2test = residual(nls_test, R2_out.solution)
    @show acc(r2train), acc(r2test)
    r2dec = plot_svm(R2_out, R2_out.solution, "r2-$(suffix)")

    @info " using LMTR to solve with" h χ
    reset!(nls_tr)
    LMTR_out = LMTR(nls_tr, h, χ, options, x0=nls_tr.meta.x0, subsolver_options = suboptions)
    lmtrtrain = residual(nls_tr, LMTR_out.solution)
    lmtrtest = residual(nls_test, LMTR_out.solution)
    nlmtr = neval_residual(nls_tr)
    nglmtr = neval_jtprod_residual(nls_tr) + neval_jprod_residual(nls_tr)
    @show acc(lmtrtrain), acc(lmtrtest)
    lmtrdec = plot_svm(LMTR_out, LMTR_out.solution, "lmtr-$(suffix)")

    #=@info " using LM to solve with" h
    reset!(nls_tr)
    LM_out = LM(nls_tr, h, options, x0=nls_tr.meta.x0, subsolver_options = suboptions)
    lmtrain = residual(nls_tr, LM_out.solution)
    lmtest = residual(nls_test, LM_out.solution)
    nlm = neval_residual(nls_tr)
    nglm = neval_jtprod_residual(nls_tr) + neval_jprod_residual(nls_tr)
    @show acc(lmtrain), acc(lmtest)
    lmdec = plot_svm(LM_out, LM_out.solution, "lm-$(suffix)")=#

    #=@info " using Sto_LM to solve with" h
    reset!(sampled_nls_tr)
    sampled_nls_tr.epoch_counter = Int[1]
    Sto_LM_out = Sto_LM(sampled_nls_tr, h, sampled_options, x0=sampled_nls_tr.meta.x0, subsolver_options = suboptions)
    
    slmtrain = residual(sampled_nls_tr, Sto_LM_out.solution)
    slmtest = residual(sampled_nls_test, Sto_LM_out.solution)
    nslm = neval_residual(sampled_nls_tr)
    ngslm = neval_jtprod_residual(sampled_nls_tr) + neval_jprod_residual(sampled_nls_tr)
    @show acc(slmtrain), acc(slmtest)
    slmdec = plot_svm(Sto_LM_out, Sto_LM_out.solution, "sto-lm-$(suffix)")=#

    @info " using Prob_LM to solve with" h

    # routine to select the output with the median accuracy on the training set
    PLM_outs = []
    plm_trains = []
    for k in 1:n_runs
        reset!(sampled_nls_tr)
        sampled_nls_tr.epoch_counter = Int[1]
        Prob_LM_out_k = Prob_LM(sampled_nls_tr, h, sampled_options, x0=sampled_nls_tr.meta.x0, subsolver_options = suboptions, version = version)
        push!(PLM_outs, Prob_LM_out_k)
        push!(plm_trains, residual(sampled_nls_tr, Prob_LM_out_k.solution))
    end
    if n_runs%2 == 1
        med_ind = (n_runs ÷ 2) + 1
    else
        med_ind = (n_runs ÷ 2)
    end
    acc_vec = acc.(plm_trains)
    sorted_acc_vec = sort(acc_vec)
    ref_value = sorted_acc_vec[med_ind]
    origin_ind = 0
    for i in eachindex(PLM_outs)
        if acc_vec[i] == ref_value
            origin_ind = i
        end
    end

    # Prob_LM_out is the run associated to the median accuracy on the training set
    Prob_LM_out = PLM_outs[origin_ind]

    plmtrain = residual(sampled_nls_tr, Prob_LM_out.solution)
    plmtest = residual(sampled_nls_test, Prob_LM_out.solution)
    nplm = neval_residual(sampled_nls_tr) / 100
    ngplm = (neval_jtprod_residual(sampled_nls_tr) + neval_jprod_residual(sampled_nls_tr)) / 100
    @show acc(plmtrain), acc(plmtest)
    plmdec = plot_svm(Prob_LM_out, Prob_LM_out.solution, "prob-lm-$version-$(suffix)")

    #=c = PGFPlots.Axis(
        [
            PGFPlots.Plots.Linear(1:length(r2dec), r2dec, mark="none", style="black, dotted", legendentry="R2"),
            PGFPlots.Plots.Linear(LM_out.solver_specific[:ResidHist], lmdec, mark="none", style="black, thick", legendentry="LM"),
            PGFPlots.Plots.Linear(LMTR_out.solver_specific[:ResidHist], lmtrdec, mark="none", style = "black, very thin", legendentry="LMTR"),
            #PGFPlots.Plots.Linear(Sto_LM_out.solver_specific[:ResidHist], slmdec, mark="none", style = "black", legendentry="Sto_LM"),
            PGFPlots.Plots.Linear(Prob_LM_out.solver_specific[:ResidHist], plmdec, mark="none", style = "black", legendentry="Sto_LM"),
        ],
        xlabel="\$ k^{th}\$   \$ f \$ Eval",
        ylabel="Objective Value",
        ymode="log",
        xmode="log",
    )
    PGFPlots.save("svm-objdec.tikz", c)=#

    temp = hcat([R2_out.solver_specific[:Fhist][end], R2_out.solver_specific[:Hhist][end],R2_out.objective, acc(r2train), acc(r2test), nr2, ngr2, sum(R2_out.solver_specific[:SubsolverCounter]), R2_out.elapsed_time],
        #[LM_out.solver_specific[:Fhist][end], LM_out.solver_specific[:Hhist][end], LM_out.objective, acc(lmtrain), acc(lmtest), nlm, nglm, sum(LM_out.solver_specific[:SubsolverCounter]), LM_out.elapsed_time],
        [LMTR_out.solver_specific[:Fhist][end], LMTR_out.solver_specific[:Hhist][end], LMTR_out.objective, acc(lmtrtrain), acc(lmtrtest), nlmtr, nglmtr, sum(LMTR_out.solver_specific[:SubsolverCounter]), LMTR_out.elapsed_time],
        #[Sto_LM_out.solver_specific[:ExactFhist][end], Sto_LM_out.solver_specific[:Hhist][end], Sto_LM_out.solver_specific[:ExactFhist][end] + Sto_LM_out.solver_specific[:Hhist][end], acc(slmtrain), acc(slmtest), nslm, ngslm, sum(Sto_LM_out.solver_specific[:SubsolverCounter]), Sto_LM_out.elapsed_time],
        [Prob_LM_out.solver_specific[:Fhist][end], Prob_LM_out.solver_specific[:Hhist][end], Prob_LM_out.objective, acc(plmtrain), acc(plmtest), nplm, ngplm, sum(Prob_LM_out.solver_specific[:SubsolverCounter]) / 100, Prob_LM_out.elapsed_time])'

    df = DataFrame(temp, [:f, :h, :fh, :x,:xt, :n, :g, :p, :s])
    T = []
    for i = 1:nrow(df)
      push!(T, Tuple(df[i, [:x, :xt]]))
    end
    select!(df, Not(:xt))
    df[!, :x] = T
    df[!, :Alg] = ["R2", "LMTR", "Prob_LM"]
    select!(df, :Alg, Not(:Alg), :)
    fmt_override = Dict(:Alg => "%s",
        :f => "%10.2f",
        :h => "%10.2f",
        :fh => "%10.2f",
        :x => "%10.2f, %10.2f",
        :n => "%i",
        :g => "%i",
        :p => "%i",
        :s => "%02.2f")
    hdr_override = Dict(:Alg => "Alg",
        :f => "\$ f \$",
        :h => "\$ h \$",
        :fh => "\$ f+h \$",
        :x => "(Train, Test)",
        :n => "\\# \$f\$",
        :g => "\\# \$ \\nabla f \$",
        :p => "\\# \$ \\prox{}\$",
        :s => "\$t \$ (s)")
    open("svm-$suffix.tex", "w") do io
        SolverBenchmark.pretty_latex_stats(io, df,
            col_formatters=fmt_override,
            hdr_override=hdr_override)
    end
end

function demo_svm_sto(;sample_rate = .05, n_runs = 1, digits = (1, 7))
    ## load phishing data from libsvm
    # A = readdlm("data_matrix.txt")
    # b = readdlm("label_vector.txt")

    # # sort into test/trainig
    # test_ind = randperm(length(b))[1:Int(floor(length(b)*.1))]
    # train_ind = setdiff(1:length(b), test_ind)
    # btest = b[test_ind]
    # Atest = A[test_ind,:]'
    # btrain = b[train_ind]
    # Atrain = A[train_ind,:]'

    nlp_train, nls_train, sol_train = RegularizedProblems.svm_train_model(digits)#Atrain, btrain) #
    nlp_test, nls_test, sol_test = RegularizedProblems.svm_test_model(digits)#Atest, btest)
    nlp_train_sto, nls_train_sto, sto_sol_train = MNIST_train_model_sto(sample_rate; digits = digits)
    nlp_test_sto, nls_test_sto, sto_sol_test = MNIST_test_model_sto(sample_rate; digits = digits)

    nlp_train = LSR1Model(nlp_train)
    λ = 1e-1
    h = RootNormLhalf(λ)
    # h = NormL0(λ)
    χ = NormLinf(1.0)

    demo_solver(nlp_train, nls_train, nls_train_sto, sol_train, nlp_test, nls_test, nls_test_sto, sol_test, h, χ, "lhalf-linf-$digits"; n_runs = n_runs)
end