function plot_Sto_LM_SVM(sample_rates::AbstractVector, versions::AbstractVector, selected_probs::AbstractVector, selected_hs::AbstractVector, selected_digits::AbstractVector; abscissa = "CPU time", n_exec = 10, smooth::Bool = false, sample_rate0::Float64 = .05, param::String = "MSE", compare::Bool = false, guide::Bool = false, MaxEpochs::Int = 1000, MaxTime = 3600.0, precision = 1e-4)
    compound = 1
    color_scheme = Dict([(1.0, 4), (.2, 5), (.1, 6), (.05, 7), (.01, 8)])
    prob_versions_names = Dict([(1, "nondec"), (2, "arbitrary"), (3, "each-it"), (4, "hybrid")])

    Confidence = Dict([("95%", 1.96), ("99%", 2.58)])
    conf = "95%"

    acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100

    for selected_prob in selected_probs
        for selected_h in selected_hs
            for digits in selected_digits
                yscale = :log10
                xscale = :log2
                gr()
                graph = Plots.plot()
                #plots of other algorithms
                if compare && (abscissa == "epoch")
                    bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound)
                    mnist_full, mnist_nls_full = RegularizedProblems.svm_train_model(digits)
                    A_ijcnn1, b_ijcnn1 = ijcnn1_load_data()
                    ijcnn1_full, ijcnn1_nls_full = RegularizedProblems.svm_model(A_ijcnn1', b_ijcnn1)
                    sampled_options_full = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = precision, ϵr = precision, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
                    subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = (selected_prob == "mnist" ? 100 : 30))

                    if selected_prob == "ijcnn1"
                        prob = ijcnn1_full
                        prob_nls = ijcnn1_nls_full
                    elseif selected_prob == "mnist"
                        prob = mnist_full
                        prob_nls = mnist_nls_full
                    end

                    x0 = ones(prob.meta.nvar)
                    m = prob_nls.nls_meta.nequ
                    l_bound = prob.meta.lvar
                    u_bound = prob.meta.uvar

                    if !smooth
                        λ = .1
                        if selected_h == "l0"
                            h = NormL0(λ)
                        elseif selected_h == "l1"
                            h = NormL1(λ)
                        elseif selected_h == "l1/2"
                            h = RootNormLhalf(λ)
                        end

                        xk_R2, k_R2, R2_out = R2(prob.f, prob.∇f!, h, sampled_options_full, x0)
                        LM_out = LM(prob_nls, h, sampled_options_full; x0 = x0, subsolver_options = subsolver_options)
                        if (h == NormL0(λ)) || (h == RootNormLhalf(λ))
                            LMTR_out = RegularizedOptimization.LMTR(prob_nls, h, NormLinf(1.0), sampled_options_full; x0 = x0, subsolver_options = subsolver_options)
                        elseif h == NormL1(λ)
                            LMTR_out = RegularizedOptimization.LMTR(prob_nls, h, NormL2(1.0), sampled_options_full; x0 = x0, subsolver_options = subsolver_options)
                        elseif h == NormL1(0.0)
                            LMTR_out = RegularizedOptimization.LMTR(prob_nls, h, NormL2(1.0), sampled_options_full; x0 = x0, subsolver_options = subsolver_options)
                        end

                        if param == "MSE"
                            plot!(1:k_R2, 0.5*(R2_out[:Fhist] + R2_out[:Hhist])/m, label = "R2", lc = :red, ls = :dashdot, xaxis = xscale, yaxis = yscale, legend=:outertopright)
                            plot!(1:length(LM_out.solver_specific[:Fhist]), 0.5*(LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist])/m, label = "LM", lc = :orange, ls = :dot, xaxis = xscale, yaxis = yscale, legend=:outertopright)
                            plot!(1:length(LMTR_out.solver_specific[:Fhist]), 0.5*(LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist])/m, label = "LMTR", lc = :black, ls=:dash, xaxis = xscale, yaxis = yscale, legend=:outertopright)
                        #elseif param == "accuracy"
                        elseif param == "objective"
                            plot!(1:k_R2, R2_out[:Fhist] + R2_out[:Hhist], label = "R2", lc = :red, ls = :dashdot, yaxis = yscale)
                            plot!(1:length(LM_out.solver_specific[:Fhist]), LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist], yaxis = yscale, label = "LM", lc = :orange, ls = :dot, legend=:outertopright)
                            plot!(1:length(LMTR_out.solver_specific[:Fhist]), LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist], yaxis = yscale, label = "LMTR", lc = :black, ls=:dash, legend=:outertopright)
                        end
                    else
                        # smooth solvers #
                        R2_out = JSOSolvers.R2(prob_nls)
                        LM_out = levenberg_marquardt(prob_nls)
                        LMTR_out = levenberg_marquardt(prob_nls, TR = true)
                        if param == "MSE"
                            plot!(1:length(R2_out.solver_specific[:Fhist]), 0.5*(R2_out.solver_specific[:Fhist] + R2_out.solver_specific[:Hhist])/m, label = "LM", lc = :orange, ls = :dot, xaxis = xscale, yaxis = yscale, legend=:outertopright)
                            plot!(1:length(LM_out.solver_specific[:Fhist]), 0.5*(LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist])/m, label = "LM", lc = :orange, ls = :dot, xaxis = xscale, yaxis = yscale, legend=:outertopright)
                            plot!(1:length(LMTR_out.solver_specific[:Fhist]), 0.5*(LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist])/m, label = "LMTR", lc = :black, ls=:dash, xaxis = xscale, yaxis = yscale, legend=:outertopright)
                        #elseif param == "accuracy"
                        elseif param == "objective"
                            plot!(1:length(R2_out.solver_specific[:Fhist]), R2_out.solver_specific[:Fhist] + R2_out.solver_specific[:Hhist], yaxis = yscale, label = "LM", lc = :orange, ls = :dot, legend=:outertopright)
                            plot!(1:length(LM_out.solver_specific[:Fhist]), LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist], yaxis = yscale, label = "LM", lc = :orange, ls = :dot, legend=:outertopright)
                            plot!(1:length(LMTR_out.solver_specific[:Fhist]), LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist], yaxis = yscale, label = "LMTR", lc = :black, ls=:dash, legend=:outertopright)
                        end
                    end
                end

                ## -------------------------------- SAMPLED VERSION ------------------------------------------ ##

                for sample_rate in sample_rates
                    nz = 10 * compound
                    #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
                    sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, σmax = 1e16, ϵa = precision, ϵr = precision, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
                    local subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = (selected_prob == "mnist" ? 100 : 30))
                    local bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound; sample_rate = sample_rate)
                    #glasso, glasso_nls, sol_glasso, g, active_groups, indset = group_lasso_model_sto(compound; sample_rate = sample_rate)
                    local ijcnn1, ijcnn1_nls, ijcnn1_sol = ijcnn1_model_sto(sample_rate)
                    #a9a, a9a_nls = a9a_model_sto(sample_rate)
                    local mnist, mnist_nls, mnist_sol = MNIST_train_model_sto(sample_rate; digits = digits)
                    #lrcomp, lrcomp_nls, sol_lrcomp = lrcomp_model(50, 20; sample_rate = sample_rate)
                    local λ = .1
                    if selected_prob == "ijcnn1"
                        nls_prob_collection = [(ijcnn1_nls, "ijcnn1-ls")]
                    elseif selected_prob == "mnist"
                        nls_prob_collection = [(mnist_nls, "mnist-train-ls")]
                    end

                    # initialize all historic collections #
                    Obj_Hists_epochs = zeros(1 + MaxEpochs, n_exec)
                    Metr_Hists_epochs = similar(Obj_Hists_epochs)
                    Time_Hists = []
                    Obj_Hists_time = []
                    for (prob, prob_name) in nls_prob_collection
                        if selected_h == "l0"
                            h = NormL0(λ)
                            h_name = "l0-norm"
                        elseif selected_h == "l1"
                            h = NormL1(λ)
                            h_name = "l1-norm"
                        elseif selected_h == "l1/2"
                            h = RootNormLhalf(λ)
                            h_name = "l1/2-norm"
                        end
                        for k in 1:n_exec
                            # executes n_exec times Sto_LM with the same inputs
                            x0 = ones(prob.meta.nvar)
                            #p = randperm(prob.meta.nvar)[1:nz]
                            #x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
                            reset!(prob)
                            try
                                if !guide
                                    SLM4_out = Sto_LM(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options)
                                else
                                    SLM4_out, Metric_hist, exact_F_hist, exact_Metric_hist, TimeHist = Sto_LM_guided(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options)
                                end
                                #reset!(prob)
                                #prob.epoch_counter = Int[1]
                                #SLM_cp_out, Metric_hist_cp, exact_F_hist_cp, exact_Metric_hist_cp, TimeHist_cp = Sto_LM_cp(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options)
                                
                                push!(Time_Hists, SLM4_out.solver_specific[:TimeHist])
                                if param == "objective"
                                    if abscissa == "epoch"
                                        @views Obj_Hists_epochs[:, k][1:length(prob.epoch_counter)] = SLM4_out.solver_specific[:ExactFhist][prob.epoch_counter]
                                        @views Obj_Hists_epochs[:, k][1:length(prob.epoch_counter)] += SLM4_out.solver_specific[:Hhist][prob.epoch_counter]
                                        @views Metr_Hists_epochs[:, k][1:length(prob.epoch_counter)] = SLM4_out.solver_specific[:ExactMetricHist][prob.epoch_counter]
                                    elseif abscissa == "CPU time"
                                        push!(Obj_Hists_time, SLM4_out.solver_specific[:ExactFhist] + SLM4_out.solver_specific[:Hhist])
                                    end
                                elseif param == "MSE"
                                    sample_size = length(prob.sample)
                                    if abscissa == "epoch"
                                        @views Obj_Hists_epochs[:, k][1:length(prob.epoch_counter)] = SLM4_out.solver_specific[:Fhist][prob.epoch_counter]
                                        @views Obj_Hists_epochs[:, k][1:length(prob.epoch_counter)] += SLM4_out.solver_specific[:Hhist][prob.epoch_counter]
                                        @views Obj_Hists_epochs[:, k][1:length(prob.epoch_counter)] ./= 2*sample_size
                                    elseif abscissa == "CPU time"
                                        push!(Obj_Hists_time, (SLM4_out.solver_specific[:Fhist] + SLM4_out.solver_specific[:Hhist]) / (2*sample_size))
                                    end
                                elseif param == "accuracy"
                                    if abscissa == "epoch"
                                        Obj_Hists_epochs[:, k] = acc.(residual.(prob, SLM4_out.solver_specific[:Xhist][prob.epoch_counter]))
                                    elseif abscissa == "CPU time"
                                        Obj_Hists_time_vec = []
                                        for i in 1:length(SLM4_out.solver_specific[:Xhist])
                                            push!(Obj_Hists_time_vec, acc(residual(prob, SLM4_out.solver_specific[:Xhist][i])))
                                        end
                                        push!(Obj_Hists_time, Obj_Hists_time_vec)
                                    end
                                end
                                if k < n_exec
                                prob.epoch_counter = Int[1]
                                end
                            catch e
                                @info "WARNING: got error" e "for run" k 
                                continue
                            end
                        end
                        if abscissa == "epoch"
                            sample_size = length(prob.sample)
                            med_obj = zeros(axes(Obj_Hists_epochs, 1))
                            std_obj = similar(med_obj)
                            med_metric = zeros(axes(Metr_Hists_epochs, 1))
                            std_metric = similar(med_metric)

                            for l in 1:length(med_obj)
                                #filter zero values if some executions fail
                                med_obj[l] = mean(filter(!iszero, Obj_Hists_epochs[l, :]))
                                std_obj[l] = std(filter(!iszero, Obj_Hists_epochs[l, :]))
                                #med_metric[l] = mean(filter(!iszero, Metr_Hists_epochs[l, :]))
                                #std_metric[l] = std(filter(!iszero, Metr_Hists_epochs[l, :]))
                            end
                            std_obj *= Confidence[conf]
                            #display(std_obj)
                            #std_metric *= Confidence[conf] / sqrt(sample_size)

                            if (param == "MSE") || (param == "accuracy") || (param == "objective")
                                if !guide
                                    plot!(axes(Obj_Hists_epochs, 1), med_obj, color=color_scheme[sample_rate], lw = 1, yaxis = yscale, label = "Sto_LM - $(sample_rate*100)%", title = "$prob_name - $n_exec runs - $digits - h = $h_name", ribbon=(std_obj, std_obj), xaxis = xscale, legend=:outertopright)
                                else
                                    if prob_name == "mnist-train-ls"
                                        plot!(axes(Obj_Hists_epochs, 1), med_obj, color=color_scheme[sample_rate], lw = 1, yaxis = yscale, label = "Sto_LM_guided - $(sample_rate*100)%", title = "$prob_name - $n_exec runs - $digits - h = $h_name", ribbon=(std_obj, std_obj), xaxis = xscale, legend=:outertopright)
                                    else
                                        plot!(axes(Obj_Hists_epochs, 1), med_obj, color=color_scheme[sample_rate], lw = 1, yaxis = yscale, label = "Sto_LM_guided - $(sample_rate*100)%", title = "$prob_name - $n_exec runs - h = $h_name", ribbon=(std_obj, std_obj), xaxis = xscale, legend=:outertopright)
                                    end
                                end
                            #plot!(axes(Obj_Hists_epochs_cp, 1), med_obj_cp, lc=color_scheme[sample_rate], lw = 1, label = "Sto_LM_cp - $(sample_rate*100)%", title = "Exact f + h for $prob_name on $n_exec runs", xaxis=:log10, yaxis=:log10, ribbon=(std_obj_cp, std_obj_cp), ls = :dot)

                            elseif param == "metric"
                                plot!(axes(Metr_Hists_epochs, 1), med_metric, color=color_scheme[sample_rate], lw = 1, yaxis = yscale, label = "Sto_LM - $(sample_rate*100)%", title = "Sampled √ξcp/νcp for $prob_name on $n_exec runs - $digits", ribbon=(std_metric, std_metric), legend=:outertopright)
                            end
                            
                        elseif abscissa == "CPU time"
                            local t = maximum(length.(Time_Hists))
                            local m = maximum(length.(Obj_Hists_time))
                            Obj_Mat_time = zeros(m, n_exec)
                            Time_mat = zeros(m, n_exec)
                            for i in 1:n_exec
                                Obj_Mat_time[:, i] .= vcat(Obj_Hists_time[i], zeros(m - length(Obj_Hists_time[i])))
                                Time_mat[:, i] .= vcat(Time_Hists[i], zeros(m - length(Time_Hists[i])))
                            end

                            sample_size = length(prob.sample)
                            med_obj = zeros(axes(Obj_Mat_time, 1))
                            std_obj = similar(med_obj)
                            #med_metric = zeros(axes(Metr_Hists_epochs, 1))
                            #std_metric = similar(med_metric)
                            med_time = zeros(axes(Time_mat, 1))

                            for l in eachindex(med_obj)
                                #filter zero values if some executions fail
                                data = filter(!iszero, Obj_Mat_time[l, :])
                                med_obj[l] = mean(data)
                                if !(param == "accuracy") && (length(data) > 1)
                                    std_obj[l] = std(data)
                                end
                                #med_metric[l] = mean(filter(!iszero, Metr_Hists_epochs[l, :]))
                                #std_metric[l] = std(filter(!iszero, Metr_Hists_epochs[l, :]))
                                med_time[l] = median(vcat(0.0, filter(!iszero, Time_mat[l, :])))
                            end
                            #=med_metric[l] = mean(filter(!iszero, Metr_Hists_epochs[l, :]))
                            std_metric[l] = std(filter(!iszero, Metr_Hists_epochs[l, :]))
                            med_time_cp[l] = mean(filter(!iszero, Time_mat_cp[l, :]))
                            end
                            std_obj *= Confidence[conf] / sqrt(sample_size)
                            std_metric *= Confidence[conf] / sqrt(sample_size)=#

                            if (param == "MSE") || (param == "accuracy") || (param == "objective")
                                if !guide
                                    plot!(sort(med_time), med_obj, color = color_scheme[sample_rate], lw = 1, yaxis = yscale, label = "Sto_LM - $(sample_rate*100)%", title = "$prob_name - $n_exec runs - $digits - h = $h_name", ribbon=(std_obj, std_obj), legend=:outertopright)
                                else
                                    if prob_name == "mnist-train-ls"
                                        plot!(sort(med_time), med_obj, color = color_scheme[sample_rate], lw = 1, yaxis = yscale, label = "Sto_LM_guided - $(sample_rate*100)%", title = "$prob_name - $n_exec runs - $digits - h = $h_name", ribbon=(std_obj, std_obj), legend=:outertopright)
                                    else
                                        plot!(sort(med_time), med_obj, color = color_scheme[sample_rate], lw = 1, yaxis = yscale, label = "Sto_LM_guided - $(sample_rate*100)%", title = "$prob_name - $n_exec runs - h = $h_name", ribbon=(std_obj, std_obj), legend=:outertopright)
                                    end
                                end

                            # cp version #
                            #plot!(sort(med_time_cp), med_obj_cp, lc = color_scheme[sample_rate], lw = 1, label = "Sto_LM_cp - $(sample_rate*100)%", title = "Exact f + h for $prob_name on $n_exec runs", xaxis=:log10, yaxis=:log10, ribbon=(std_obj_cp, std_obj_cp), ls = :dot)
                            elseif param == "metric"
                            plot!(axes(Metr_Hists, 1), med_metric, color = color_scheme[sample_rate], lw = 1, yaxis = yscale, label = "Sto_LM - $(sample_rate*100)%", title = "Sampled √ξcp/νcp for $prob_name on $n_exec runs - $digits", ribbon=(std_metric, std_metric), ls = :dot, legend=:outertopright)
                            end
                        end
                        prob.epoch_counter = Int[1]
                    end
                end

                ## ---------------------- DYNAMIC SAMPLE RATE APPROACH ------------------------------ ##

                for version in versions
                    nz = 10 * compound
                    #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
                    sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, σmax = 1e16, ϵa = precision, ϵr = precision, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
                    local subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = (selected_prob == "mnist" ? 100 : 30))
                    local bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound; sample_rate = sample_rate0)
                    #glasso, glasso_nls, sol_glasso, g, active_groups, indset = group_lasso_model_sto(compound; sample_rate = sample_rate)
                    local ijcnn1, ijcnn1_nls, ijcnn1_sol = ijcnn1_model_sto(sample_rate0)
                    #a9a, a9a_nls = a9a_model_sto(sample_rate)
                    local mnist, mnist_nls, mnist_sol = MNIST_train_model_sto(sample_rate0; digits = digits)
                    #lrcomp, lrcomp_nls, sol_lrcomp = lrcomp_model(50, 20; sample_rate = sample_rate)
                    local λ = .1
                    if selected_prob == "ijcnn1"
                        nls_prob_collection = [(ijcnn1_nls, "ijcnn1-ls")]
                    elseif selected_prob == "mnist"
                        nls_prob_collection = [(mnist_nls, "mnist-train-ls")]
                    end
                    Obj_Hists_epochs_prob = zeros(1 + MaxEpochs, n_exec)
                    Metr_Hists_epochs_prob = similar(Obj_Hists_epochs_prob)
                    Time_Hists_prob = []
                    Obj_Hists_time_prob = []

                    for (prob, prob_name) in nls_prob_collection
                        if selected_h == "l0"
                            h = NormL0(λ)
                            h_name = "l0-norm"
                        elseif selected_h == "l1"
                            h = NormL1(λ)
                            h_name = "l1-norm"
                        elseif selected_h == "l1/2"
                            h = RootNormLhalf(λ)
                            h_name = "l1/2-norm"
                        end
                        for k in 1:n_exec
                            # executes n_exec times Sto_LM with the same inputs
                            x0 = ones(prob.meta.nvar)
                            #p = randperm(prob.meta.nvar)[1:nz]
                            #x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
                            reset!(prob)
                            #try
                            PLM_out = Prob_LM(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options, sample_rate0 = sample_rate0, version = version)

                            push!(Time_Hists_prob, PLM_out.solver_specific[:TimeHist])
                            if param == "objective"
                                if abscissa == "epoch"
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:ExactFhist][prob.epoch_counter]
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] += PLM_out.solver_specific[:Hhist][prob.epoch_counter]
                                    @views Metr_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:ExactMetricHist][prob.epoch_counter]
                                elseif abscissa == "CPU time"
                                    push!(Obj_Hists_time_prob, PLM_out.solver_specific[:ExactFhist] + PLM_out.solver_specific[:Hhist])
                                end
                            elseif param == "MSE"
                                if abscissa == "epoch"
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] = PLM_out.solver_specific[:Fhist][prob.epoch_counter]
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] += PLM_out.solver_specific[:Hhist][prob.epoch_counter]
                                    @views Obj_Hists_epochs_prob[:, k][1:length(prob.epoch_counter)] ./= ceil.(2 * prob.nls_meta.nequ * PLM_out.solver_specific[:SampleRateHist][prob.epoch_counter])
                                elseif abscissa == "CPU time"
                                    Obj_Hists_time_prob_vec = PLM_out.solver_specific[:Fhist] + PLM_out.solver_specific[:Hhist]
                                    Obj_Hists_time_prob_vec ./= ceil.(2 * prob.nls_meta.nequ * PLM_out.solver_specific[:SampleRateHist])
                                    push!(Obj_Hists_time_prob, Obj_Hists_time_prob_vec)
                                end
                            elseif param == "accuracy"
                                if abscissa == "epoch"
                                    Obj_Hists_epochs[:, k] = acc.(residual.(prob, PLM_out.solver_specific[:Xhist][prob.epoch_counter]))
                                elseif abscissa == "CPU time"
                                    Obj_Hists_time_vec_prob = []
                                    for i in 1:length(PLM_out.solver_specific[:Xhist])
                                        push!(Obj_Hists_time_vec_prob, acc(residual(prob, PLM_out.solver_specific[:Xhist][i])))
                                    end
                                    push!(Obj_Hists_time_prob, Obj_Hists_time_vec_prob)
                                end
                            end
                            if k < n_exec
                            prob.epoch_counter = Int[1]
                            end
                        end
                        if abscissa == "epoch"
                            med_obj_prob = zeros(axes(Obj_Hists_epochs_prob, 1))
                            std_obj_prob = similar(med_obj_prob)
                            for l in 1:length(med_obj_prob)
                                med_obj_prob[l] = mean(filter(!iszero, Obj_Hists_epochs_prob[l, :]))
                                std_obj_prob[l] = std(filter(!iszero, Obj_Hists_epochs_prob[l, :]))
                            end
                            std_obj_prob *= Confidence[conf]
                            replace!(std_obj_prob, NaN=>0.0)
                            #std_metric *= Confidence[conf] / sqrt(sample_size)

                            if (param == "MSE") || (param == "accuracy") || (param == "objective")
                                if prob_name == "mnist-train-ls"
                                    plot!(1:length(med_obj_prob), med_obj_prob, color = version, lw = 1, yaxis = yscale, label = "PLM - $(prob_versions_names[version])", title = "$prob_name - $n_exec runs - $digits - h = $h_name", ribbon=(std_obj_prob, std_obj_prob), xaxis = xscale, legend=:outertopright)
                                else
                                    plot!(1:length(med_obj_prob), med_obj_prob, color = version, lw = 1, yaxis = yscale, label = "PLM - $(prob_versions_names[version])", title = "$prob_name - $n_exec runs - h = $h_name", ribbon=(std_obj_prob, std_obj_prob), xaxis = xscale, legend=:outertopright)
                                end
                            elseif param == "metric"
                                plot!(axes(Metr_Hists_epochs, 1), med_metric, color = version, lw = 1, yaxis = yscale, label = "PLM - $(prob_versions_names[version])", title = "Sampled √ξcp/νcp for $prob_name on $n_exec runs - $digits", ribbon=(std_metric, std_metric), xaxis = xscale, legend=:outertopright)
                            end
                            
                        elseif abscissa == "CPU time"
                            local t_prob = maximum(length.(Time_Hists_prob))
                            local m_prob = maximum(length.(Obj_Hists_time_prob))
                            Obj_Mat_time_prob = zeros(m_prob, n_exec)
                            Time_mat_prob = zeros(t_prob, n_exec)
                            for i in 1:n_exec
                                Obj_Mat_time_prob[:, i] .= vcat(Obj_Hists_time_prob[i], zeros(m_prob - length(Obj_Hists_time_prob[i])))
                                Time_mat_prob[:, i] .= vcat(Time_Hists_prob[i], zeros(m_prob - length(Time_Hists_prob[i])))
                            end

                            med_obj_prob = zeros(axes(Obj_Mat_time_prob, 1))
                            std_obj_prob = similar(med_obj_prob)
                            med_time_prob = zeros(axes(Time_mat_prob, 1))

                            for l in 1:length(med_obj_prob)
                                data_prob = filter(!iszero, Obj_Mat_time_prob[l, :])
                                med_obj_prob[l] = mean(data_prob)
                                if !(param == "accuracy") && (length(data_prob) > 1)
                                    std_obj_prob[l] = std(data_prob)
                                end
                                med_time_prob[l] = median(vcat(0.0, filter(!iszero, Time_mat_prob[l, :])))
                            end
                            #=med_metric[l] = mean(filter(!iszero, Metr_Hists_epochs[l, :]))
                            std_metric[l] = std(filter(!iszero, Metr_Hists_epochs[l, :]))
                            med_time_cp[l] = mean(filter(!iszero, Time_mat_cp[l, :]))
                            end
                            std_obj *= Confidence[conf] / sqrt(sample_size)
                            std_metric *= Confidence[conf] / sqrt(sample_size)=#

                            if (param == "MSE") || (param == "accuracy") || (param == "objective")
                                if prob_name == "mnist-train-ls"
                                    plot!(sort(med_time_prob), med_obj_prob, color = version, lw = 1, yaxis = yscale, label = "PLM - $(prob_versions_names[version])", title = "$prob_name - $n_exec runs - $digits - h = $h_name", ribbon=(std_obj_prob, std_obj_prob), legend=:outertopright)
                                else
                                    plot!(sort(med_time_prob), med_obj_prob, color = version, lw = 1, yaxis = yscale, label = "PLM - $(prob_versions_names[version])", title = "$prob_name - $n_exec runs - h = $h_name", ribbon=(std_obj_prob, std_obj_prob), legend=:outertopright)
                                end    
                                # cp version #
                            #plot!(sort(med_time_cp), med_obj_cp, lc = color_scheme[sample_rate], lw = 1, label = "Sto_LM_cp - $(sample_rate*100)%", title = "Exact f + h for $prob_name on $n_exec runs", xaxis=:log10, yaxis=:log10, ribbon=(std_obj_cp, std_obj_cp), ls = :dot)
                            elseif param == "metric"
                                plot!(axes(Metr_Hists, 1), med_metric, color = version, lw = 1, yaxis = yscale, label = "PLM - $(prob_versions_names[version])", title = "Sampled √ξcp/νcp for $prob_name on $n_exec runs - $digits", ribbon=(std_metric, std_metric), ls = :dot, legend=:outertopright)
                            end
                        end
                        prob.epoch_counter = Int[1]
                    end
                end

                if abscissa == "CPU time"
                    xlabel!("CPU time [s]")
                else
                    xlabel!(abscissa)
                end

                ylabel!(param)
                display(graph)
            end
        end
    end
end