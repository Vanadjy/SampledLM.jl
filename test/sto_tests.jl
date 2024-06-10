compound = 1
#sample_rates = [1.0, .2, .1, .05, .01]
#sample_rates = [1.0, .2, .1, .05]
sample_rates = [.05]
color_scheme = Dict([(1.0, 1), (.2, 2), (.1, 3), (.05, 4), (.01, 5)])

#sample_rates = [.2]

plot_parameter = ["objective", "metric", "MSE"]
param = plot_parameter[3]

abscissas = ["epoch", "CPU time"]
abscissa = abscissas[1]
sampled_res = true
Confidence = Dict([("95%", 1.96), ("99%", 2.58)])
conf = "95%"
n_exec = 10
guide = false
probabilist = false

MaxEpochs = 0
MaxTime = 0.0
if abscissa == "epoch"
  MaxEpochs = 20
  MaxTime = 3600.0
elseif abscissa == "CPU time"
  MaxEpochs = 1000
  MaxTime = 10.0
end

#selected_probs = ["ijcnn1", "mnist"]
selected_probs = ["mnist"]
#selected_hs = ["l0", "l1"]
selected_hs = ["l1"]
#selected_prob = "mnist"

acc = vec -> length(findall(x -> x < 1, vec)) / length(vec) * 100

for selected_prob in selected_probs
  for selected_h in selected_hs
    # -- SAMPLED VERSION -- #

    for sample_rate in sample_rates
      nz = 10 * compound
      #options = RegularizedOptimization.ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10, spectral = true)
      sampled_options = ROSolverOptions(η3 = .4, ν = 1.0, νcp = 2.0, β = 1e16, ϵa = 1e-3, ϵr = 1e-3, verbose = 10, maxIter = MaxEpochs, maxTime = MaxTime;)
      local subsolver_options = RegularizedOptimization.ROSolverOptions(ϵa = 1e-2, maxIter = 15)
      local bpdn, bpdn_nls, sol_bpdn = bpdn_model_sto(compound; sample_rate = sample_rate)
      #glasso, glasso_nls, sol_glasso, g, active_groups, indset = group_lasso_model_sto(compound; sample_rate = sample_rate)
      ijcnn1, ijcnn1_nls = ijcnn1_model_sto(sample_rate)
      #a9a, a9a_nls = a9a_model_sto(sample_rate)
      mnist, mnist_nls = MNIST_train_model_sto(sample_rate)
      #lrcomp, lrcomp_nls, sol_lrcomp = lrcomp_model(50, 20; sample_rate = sample_rate)
      local λ = .1
      if selected_prob == "ijcnn1"
        nls_prob_collection = [(ijcnn1_nls, "ijcnn1-ls")]
      elseif selected_prob == "mnist"
        nls_prob_collection = [(mnist_nls, "mnist-train-ls")]
      end

      for (prob, prob_name) in nls_prob_collection
        if selected_h == "l0"
          h = NormL0(λ)
          h_name = "l0-norm"
        elseif selected_h == "l1"
          h = NormL1(λ)
          h_name = "l1-norm"
        end
          for k in 1:n_exec
            # executes n_exec times Sto_LM with the same inputs
            @testset "$prob_name-Sto_LM-$(h_name) - τ = $(sample_rate*100)%" begin
              x0 = ones(prob.meta.nvar)
              #p = randperm(prob.meta.nvar)[1:nz]
              #x0[p[1:nz]] = sign.(randn(nz))  # initial guess with nz nonzeros (necessary for h = B0)
              reset!(prob)
              if !probabilist
                if !guide
                  SLM4_out, Metric_hist, exact_F_hist, exact_Metric_hist, TimeHist = Sto_LM(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options)
                else
                  SLM4_out, Metric_hist, exact_F_hist, exact_Metric_hist, TimeHist = Sto_LM_guided(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options)
                end
              else
                prob.sample_rate = .05
                SLM4_out, Metric_hist, exact_F_hist, exact_Metric_hist, TimeHist = Prob_LM(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options)
              end
              #reset!(prob)
              #prob.epoch_counter = Int[1]
              #SLM_cp_out, Metric_hist_cp, exact_F_hist_cp, exact_Metric_hist_cp, TimeHist_cp = Sto_LM_cp(prob, h, sampled_options; x0 = x0, subsolver_options = subsolver_options)

              @test typeof(SLM4_out.solution) == typeof(prob.meta.x0)
              @test length(SLM4_out.solution) == prob.meta.nvar
              @test typeof(SLM4_out.solver_specific[:Fhist]) == typeof(SLM4_out.solution)
              @test typeof(SLM4_out.solver_specific[:Hhist]) == typeof(SLM4_out.solution)
              @test typeof(SLM4_out.solver_specific[:SubsolverCounter]) == Array{Int, 1}
              @test typeof(SLM4_out.dual_feas) == eltype(SLM4_out.solution)
              @test length(SLM4_out.solver_specific[:Fhist]) == length(SLM4_out.solver_specific[:Hhist])
              @test length(SLM4_out.solver_specific[:Fhist]) ==
                      length(SLM4_out.solver_specific[:SubsolverCounter])
              @test length(SLM4_out.solver_specific[:Fhist]) == length(SLM4_out.solver_specific[:NLSGradHist])
              @test SLM4_out.solver_specific[:NLSGradHist][end] ==
                prob.counters.neval_jprod_residual + prob.counters.neval_jtprod_residual - 1
                #@test obj(prob, SLM4_out.solution) == SLM4_out.solver_specific[:Fhist][end]
              @test h(SLM4_out.solution) == SLM4_out.solver_specific[:Hhist][end]
              @test SLM4_out.status == :max_iter
            end
          end
        end
      end


          #=if abscissa == "epoch"
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

            if (param == "MSE") || (param == "objective")
              if !guide
                plot!(axes(Obj_Hists_epochs, 1), med_obj, lc=color_scheme[sample_rate], lw = 1, linetype=:steppre, xaxis = :log2,  yaxis = yscale, label = "Sto_LM - $(sample_rate*100)%", title = "$prob_name on $n_exec runs - h = $h_name", ribbon=(std_obj, std_obj))
              else
                plot!(axes(Obj_Hists_epochs, 1), med_obj, lc=color_scheme[sample_rate], lw = 1, linetype=:steppre, xaxis = :log2,  yaxis = yscale, label = "Sto_LM_guided - $(sample_rate*100)%", title = "$prob_name on $n_exec runs - h = $h_name", ribbon=(std_obj, std_obj))
              end
              #plot!(axes(Obj_Hists_epochs_cp, 1), med_obj_cp, lc=color_scheme[sample_rate], lw = 1, label = "Sto_LM_cp - $(sample_rate*100)%", title = "Exact f + h for $prob_name on $n_exec runs", xaxis=:log10, yaxis=:log10, ribbon=(std_obj_cp, std_obj_cp), ls = :dot)

            elseif param == "metric"
              plot!(axes(Metr_Hists_epochs, 1), med_metric, lc=color_scheme[sample_rate], lw = 1, linetype=:steppre, label = "Sto_LM - $(sample_rate*100)%", title = "Sampled √ξcp/νcp for $prob_name on $n_exec runs", xaxis=:log10, ribbon=(std_metric, std_metric))
            end
            
          elseif abscissa == "CPU time"
            local t = maximum(length.(Time_Hists))
            local m = maximum(length.(Obj_Hists_time))
            Obj_Mat_time = zeros(m, n_exec)
            Time_mat = zeros(t, n_exec)

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
              if length(data) > 1
                std_obj[l] = std(data)
              end
              #med_metric[l] = mean(filter(!iszero, Metr_Hists_epochs[l, :]))
              #std_metric[l] = std(filter(!iszero, Metr_Hists_epochs[l, :]))
              med_time[l] = mean(vcat(0.0, filter(!iszero, Time_mat[l, :])))
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

            if (param == "MSE") || (param == "objective")
              if !guide
                plot!(sort(med_time), med_obj, lc = color_scheme[sample_rate], lw = 1, linetype=:steppre, label = "Sto_LM - $(sample_rate*100)%", title = "$prob_name on $n_exec runs - h = $h_name", ribbon=(std_obj, std_obj))
              else
                plot!(sort(med_time), med_obj, lc = color_scheme[sample_rate], lw = 1, linetype=:steppre, label = "Sto_LM_guided - $(sample_rate*100)%", title = "$prob_name on $n_exec runs - h = $h_name", ribbon=(std_obj, std_obj))
              end

              # cp version #
              #plot!(sort(med_time_cp), med_obj_cp, lc = color_scheme[sample_rate], lw = 1, label = "Sto_LM_cp - $(sample_rate*100)%", title = "Exact f + h for $prob_name on $n_exec runs", xaxis=:log10, yaxis=:log10, ribbon=(std_obj_cp, std_obj_cp), ls = :dot)
            elseif param == "metric"
              plot!(axes(Metr_Hists, 1), med_metric, lc = color_scheme[sample_rate], lw = 1, linetype=:steppre, label = "Sto_LM - $(sample_rate*100)%", title = "Sampled √ξcp/νcp for $prob_name on $n_exec runs", ribbon=(std_metric, std_metric), ls = :dot)
            end
          end
      end
    end

    if abscissa == "CPU time"
      xlabel!("CPU time [s]")
    else
      xlabel!(abscissa)
    end

    if param == "objective"
      ylabel!("f + h")
    elseif param == "metric"
      ylabel!("√ξcp/νcp")
    elseif param == "MSE"
      ylabel!("MSE")
    end
    display(graph)
  end
end=#