function plot_Sto_LM(sample_rates::AbstractVector, selected_probs::AbstractVector, selected_hs::AbstractVector; abscissa = "CPU time", n_exec = 10, smooth::Bool = false, sample_rate0::Float64 = .05)
    compound = 1
    color_scheme = Dict([(1.0, 1), (.2, 2), (.1, 3), (.05, 4), (.01, 5)])

    plot_parameter = ["objective", "metric", "MSE", "accuracy"]
    param = plot_parameter[1]

    Confidence = Dict([("95%", 1.96), ("99%", 2.58)])
    conf = "95%"
    guide = false
    compare = true
    probabilist = true

    MaxEpochs = 0
    MaxTime = 0.0
    if abscissa == "epoch"
    MaxEpochs = 20
    MaxTime = 3600.0
    elseif abscissa == "CPU time"
    MaxEpochs = 1000
    MaxTime = 10.0
    end

    acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100

    for selected_prob in selected_probs
        for selected_h in selected_hs
            for version in 3:3
                yscale = :log10
                xscale = :log2
                gr()
                graph = plot()
                #plots of other algorithms
                if compare && (abscissa == "epoch")
                    bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound)
                    mnist_full, mnist_nls_full = RegularizedProblems.svm_train_model()
                    A_ijcnn1, b_ijcnn1 = ijcnn1_load_data()
                    ijcnn1_full, ijcnn1_nls_full = RegularizedProblems.svm_model(A_ijcnn1', b_ijcnn1)
                    sampled_options_full = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-16, ϵr = 1e-16, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
                    subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = 15)

                    if selected_prob == "ijcnn1"
                        prob = ijcnn1_full
                        prob_nls = ijcnn1_nls_full
                    elseif selected_prob == "mnist"
                        prob = mnist_full
                        prob_nls = mnist_nls_full
                    end

                    λ = .1
                    if !smooth
                        if selected_h == "l0"
                            h = NormL0(λ)
                        elseif selected_h == "l1"
                            h = NormL1(λ)
                        elseif selected_h == "l1/2"
                            h = RootNormLhalf(λ)
                        end
                    else
                        h = NormL1(0.0)
                    end

                    x0 = ones(prob.meta.nvar)
                    m = prob_nls.nls_meta.nequ
                    l_bound = prob.meta.lvar
                    u_bound = prob.meta.uvar

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
                        plot!(1:k_R2, 0.5*(R2_out[:Fhist] + R2_out[:Hhist])/m, label = "R2", lc = :red, ls = :dashdot, linetype=:steppre, xaxis = xscale, yaxis = yscale)
                        plot!(1:length(LM_out.solver_specific[:Fhist]), 0.5*(LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist])/m, xaxis = xscale, yaxis = yscale, linetype=:steppre, label = "LM", lc = :orange, ls = :dot)
                        plot!(1:length(LMTR_out.solver_specific[:Fhist]), 0.5*(LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist])/m, xaxis = xscale, yaxis = yscale, linetype=:steppre, label = "LMTR", lc = :black, ls=:dash)
                    #elseif param == "accuracy"
                    elseif param == "objective"
                        plot!(1:k_R2, R2_out[:Fhist] + R2_out[:Hhist], label = "R2", lc = :red, ls = :dashdot, linetype=:steppre, xaxis = xscale, yaxis = yscale)
                        plot!(1:length(LM_out.solver_specific[:Fhist]), LM_out.solver_specific[:Fhist] + LM_out.solver_specific[:Hhist], xaxis = xscale, yaxis = yscale, linetype=:steppre, label = "LM", lc = :orange, ls = :dot)
                        plot!(1:length(LMTR_out.solver_specific[:Fhist]), LMTR_out.solver_specific[:Fhist] + LMTR_out.solver_specific[:Hhist], xaxis = xscale, yaxis = yscale, linetype=:steppre, label = "LMTR", lc = :black, ls=:dash)
                    end
                end

                # -------------------------------- SAMPLED VERSION ------------------------------------------ #

                for sample_rate in sample_rates
                    nz = 10 * compound
                    #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
                    sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, ϵa = 1e-4, ϵr = 1e-4, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
                    local subsolver_options = RegularizedOptimization.ROSolverOptions(maxIter = 15)
                    local bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound; sample_rate = sample_rate)
                    #glasso, glasso_nls, sol_glasso, g, active_groups, indset = group_lasso_model_sto(compound; sample_rate = sample_rate)
                    local ijcnn1, ijcnn1_nls, ijcnn1_sol = ijcnn1_model_sto(sample_rate)
                    #a9a, a9a_nls = a9a_model_sto(sample_rate)
                    local mnist, mnist_nls, mnist_sol = MNIST_train_model_sto(sample_rate)
                    #lrcomp, lrcomp_nls, sol_lrcomp = lrcomp_model(50, 20; sample_rate = sample_rate)
                    local λ = .1
                    if selected_prob == "ijcnn1"
                        nls_prob_collection = [(ijcnn1_nls, "ijcnn1-ls")]
                    elseif selected_prob == "mnist"
                        nls_prob_collection = [(mnist_nls, "mnist-train-ls")]
                    end

                    # initialize all historic collections #
                    Obj_Hists_epochs = zeros(1 + MaxEpochs, n_exec)
                    Obj_Hists_epochs_cp = similar(Obj_Hists_epochs)
                    Metr_Hists_epochs = similar(Obj_Hists_epochs)
                    Metr_Hists_epochs_cp = similar(Obj_Hists_epochs)
                    Time_Hists = []
                    Time_Hists_cp = []
                    Obj_Hists_time = []
                    Obj_Hists_time_cp = []
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
                                if probabilist
                                    if !guide
                                        SLM4_out = Sto_LM(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options)
                                    else
                                        SLM4_out, Metric_hist, exact_F_hist, exact_Metric_hist, TimeHist = Sto_LM_guided(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options)
                                    end
                                elseif probabilist
                                    SLM4_out = Prob_LM(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options, sample_rate0 = sample_rate0, version = version)
                                end
                                #reset!(prob)
                                #prob.epoch_counter = Int[1]
                                #SLM_cp_out, Metric_hist_cp, exact_F_hist_cp, exact_Metric_hist_cp, TimeHist_cp = Sto_LM_cp(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options)
                                
                                push!(Time_Hists, SLM4_out.solver_specific[:TimeHist])
                                if param == "objective"
                                    if abscissa == "epoch"
                                        Obj_Hists_epochs[:, k] = SLM4_out.solver_specific[:ExactFhist][prob.epoch_counter]
                                        Obj_Hists_epochs[:, k] += SLM4_out.solver_specific[:Hhist][prob.epoch_counter]
                                        Metr_Hists_epochs[:, k] = SLM4_out.solver_specific[:ExactMetricHist][prob.epoch_counter]

                                        # cp version #
                                        #=Obj_Hists_epochs_cp[:, k] = exact_F_hist_cp[prob.epoch_counter]
                                        Obj_Hists_epochs_cp[:, k] += SLM_cp_out.solver_specific[:Hhist][prob.epoch_counter]
                                        Metr_Hists_epochs_cp[:, k] = exact_Metric_hist_cp[prob.epoch_counter]=#
                                    elseif abscissa == "CPU time"
                                        push!(Obj_Hists_time, SLM4_out.solver_specific[:ExactFhist] + SLM4_out.solver_specific[:Hhist])

                                        # cp version #
                                        #push!(Obj_Hists_time_cp, exact_F_hist_cp + SLM_cp_out.solver_specific[:Hhist])
                                    end
                                elseif param == "MSE"
                                    sample_size = length(prob.sample)
                                    if abscissa == "epoch"
                                        Obj_Hists_epochs[:, k] = SLM4_out.solver_specific[:Fhist][prob.epoch_counter]
                                        Obj_Hists_epochs[:, k] += SLM4_out.solver_specific[:Hhist][prob.epoch_counter]
                                        Obj_Hists_epochs[:, k] ./= 2*sample_size
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
                            #=catch e
                                k -= 1
                                continue
                            end=#
                        end
                        if abscissa == "epoch"
                            sample_size = length(prob.sample)
                            med_obj = zeros(axes(Obj_Hists_epochs, 1))
                            std_obj = similar(med_obj)
                            med_metric = zeros(axes(Metr_Hists_epochs, 1))
                            std_metric = similar(med_metric)

                            # cp version #
                            #med_obj_cp = zeros(axes(Obj_Hists_epochs, 1))
                            #std_obj_cp = similar(med_obj)
                            for l in 1:length(med_obj)
                            #filter zero values if some executions fail
                            med_obj[l] = mean(filter(!iszero, Obj_Hists_epochs[l, :]))
                            std_obj[l] = std(filter(!iszero, Obj_Hists_epochs[l, :]))
                            #med_metric[l] = mean(filter(!iszero, Metr_Hists_epochs[l, :]))
                            #std_metric[l] = std(filter(!iszero, Metr_Hists_epochs[l, :]))

                            # cp version #
                            #med_obj_cp[l] = mean(filter(!iszero, Obj_Hists_epochs_cp[l, :]))
                            #std_obj_cp[l] = std(filter(!iszero, Obj_Hists_epochs_cp[l, :]))
                            end
                            std_obj *= Confidence[conf]
                            #display(std_obj)
                            #std_metric *= Confidence[conf] / sqrt(sample_size)

                            # cp version #
                            #std_obj_cp *= Confidence[conf] / sqrt(sample_size)

                            if (param == "MSE") || (param == "accuracy") || (param == "objective")
                                if !probabilist
                                    if !guide
                                        plot!(axes(Obj_Hists_epochs, 1), med_obj, color=color_scheme[sample_rate], lw = 1, linetype=:steppre, xaxis = xscale, yaxis = yscale, label = "Sto_LM - $(sample_rate*100)%", title = "$prob_name on $n_exec runs - h = $h_name", ribbon=(std_obj, std_obj))
                                    else
                                        plot!(axes(Obj_Hists_epochs, 1), med_obj, color=color_scheme[sample_rate], lw = 1, linetype=:steppre, xaxis = xscale, yaxis = yscale, label = "Sto_LM_guided - $(sample_rate*100)%", title = "$prob_name on $n_exec runs - h = $h_name", ribbon=(std_obj, std_obj))
                                    end
                                else
                                    plot!(axes(Obj_Hists_epochs, 1), med_obj, color = color_scheme[sample_rate], lw = 1, linetype=:steppre, label = "Prob_LM - version $version", title = "$prob_name on $n_exec runs - h = $h_name", ribbon=(std_obj, std_obj))
                                end
                            #plot!(axes(Obj_Hists_epochs_cp, 1), med_obj_cp, lc=color_scheme[sample_rate], lw = 1, label = "Sto_LM_cp - $(sample_rate*100)%", title = "Exact f + h for $prob_name on $n_exec runs", xaxis=:log10, yaxis=:log10, ribbon=(std_obj_cp, std_obj_cp), ls = :dot)

                            elseif param == "metric"
                                plot!(axes(Metr_Hists_epochs, 1), med_metric, color=color_scheme[sample_rate], lw = 1, linetype=:steppre, label = "Sto_LM - version $version", title = "Sampled √ξcp/νcp for $prob_name on $n_exec runs", ribbon=(std_metric, std_metric))
                            end
                            
                        elseif abscissa == "CPU time"
                            local t = maximum(length.(Time_Hists))
                            local m = maximum(length.(Obj_Hists_time))
                            Obj_Mat_time = zeros(m, n_exec)
                            Time_mat = zeros(m, n_exec)

                            # cp version #
                            #=local t = maximum(length.(Time_Hists_cp))
                            local m = maximum(length.(Obj_Hists_time_cp))
                            Obj_Mat_time_cp = zeros(m, n_exec)
                            Time_mat_cp = zeros(t, n_exec)=#
                            for i in 1:n_exec
                                Obj_Mat_time[:, i] .= vcat(Obj_Hists_time[i], zeros(m - length(Obj_Hists_time[i])))
                                Time_mat[:, i] .= vcat(Time_Hists[i], zeros(m - length(Time_Hists[i])))

                                # cp version #
                                #Obj_Mat_time_cp[:, i] .= vcat(Obj_Hists_time_cp[i], zeros(m - length(Obj_Hists_time_cp[i])))
                                #Time_mat_cp[:, i] .= vcat(Time_Hists_cp[i], zeros(m - length(Time_Hists_cp[i])))
                            end
                            sample_size = length(prob.sample)
                            med_obj = zeros(axes(Obj_Mat_time, 1))
                            std_obj = similar(med_obj)
                            #med_metric = zeros(axes(Metr_Hists_epochs, 1))
                            #std_metric = similar(med_metric)
                            med_time = zeros(axes(Time_mat, 1))

                            #=sample_size_cp = length(prob.sample)
                            med_obj_cp = zeros(axes(Obj_Mat_time_cp, 1))
                            std_obj_cp = similar(med_obj_cp)
                            med_time_cp = zeros(axes(Time_mat_cp, 1))=#

                            for l in 1:length(med_obj)
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

                            # cp version #
                            #=for l in 1:length(med_obj_cp)
                            #filter zero values if some executions fail
                            data_cp = filter(!iszero, Obj_Mat_time_cp[l, :])
                            med_obj_cp[l] = mean(data_cp)
                            if length(data_cp) > 1
                                std_obj_cp[l] = std(data_cp)
                            end
                            #med_metric[l] = mean(filter(!iszero, Metr_Hists_epochs[l, :]))
                            #std_metric[l] = std(filter(!iszero, Metr_Hists_epochs[l, :]))
                            med_time_cp[l] = mean(filter(!iszero, Time_mat_cp[l, :]))
                            end=#
                            #std_obj *= Confidence[conf] / sqrt(sample_size)
                            #std_metric *= Confidence[conf] / sqrt(sample_size)

                            if (param == "MSE") || (param == "accuracy") || (param == "objective")
                                if !probabilist
                                    if !guide
                                        plot!(sort(med_time), med_obj, color = color_scheme[sample_rate], lw = 1, linetype=:steppre, label = "Sto_LM - $(sample_rate*100)%", title = "$prob_name on $n_exec runs - h = $h_name", ribbon=(std_obj, std_obj))
                                    else
                                        plot!(sort(med_time), med_obj, color = color_scheme[sample_rate], lw = 1, linetype=:steppre, label = "Sto_LM_guided - $(sample_rate*100)%", title = "$prob_name on $n_exec runs - h = $h_name", ribbon=(std_obj, std_obj))
                                    end
                                else
                                    plot!(sort(med_time), med_obj, color = color_scheme[sample_rate], lw = 1, linetype=:steppre, label = "Prob_LM - $(sample_rate0*100)%", title = "$prob_name on $n_exec runs - h = $h_name", ribbon=(std_obj, std_obj))
                                end

                            # cp version #
                            #plot!(sort(med_time_cp), med_obj_cp, lc = color_scheme[sample_rate], lw = 1, label = "Sto_LM_cp - $(sample_rate*100)%", title = "Exact f + h for $prob_name on $n_exec runs", xaxis=:log10, yaxis=:log10, ribbon=(std_obj_cp, std_obj_cp), ls = :dot)
                            elseif param == "metric"
                            plot!(axes(Metr_Hists, 1), med_metric, color = color_scheme[sample_rate], lw = 1, linetype=:steppre, label = "Sto_LM - $(sample_rate*100)%", title = "Sampled √ξcp/νcp for $prob_name on $n_exec runs", ribbon=(std_metric, std_metric), ls = :dot)
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