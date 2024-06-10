"""
    Prob_LM(nls, h, options; kwargs...)

A Levenberg-Marquardt method for the problem

    min ½ ‖F(x)‖² + h(x)

where F: ℝⁿ → ℝᵐ and its Jacobian J are Lipschitz continuous and h: ℝⁿ → ℝ is
lower semi-continuous, proper and prox-bounded.

At each iteration, a step s is computed as an approximate solution of

    min  ½ ‖J(x) s + F(x)‖² + ½ σ ‖s‖² + ψ(s; x)

where F(x) and J(x) are the residual and its Jacobian at x, respectively, ψ(s; x) = h(x + s),
and σ > 0 is a regularization parameter.

In this version of the algorithm, the smooth part of both the objective and the model are estimations as 
the quantities are sampled ones from the original data of the Problem.

### Arguments

* `nls::SampledBAModel`: a smooth nonlinear least-squares problem associated to a Bundle Adjustment Problem
* `h`: a regularizer such as those defined in ProximalOperators
* `options::ROSolverOptions`: a structure containing algorithmic parameters

### Keyword arguments

* `x0::AbstractVector`: an initial guess (default: `nls.meta.x0`)
* `subsolver_logger::AbstractLogger`: a logger to pass to the subproblem solver
* `subsolver`: the procedure used to compute a step (`PG` or `R2`)
* `subsolver_options::ROSolverOptions`: default options to pass to the subsolver.
* `selected::AbstractVector{<:Integer}`: list of selected indexes for the sampling 

### Return values

* `xk`: the final iterate
* `Fobj_hist`: an array with the history of values of the smooth objective
* `Hobj_hist`: an array with the history of values of the nonsmooth objective
* `Complex_hist`: an array with the history of number of inner iterations.
"""
function Prob_LM(
  nls::SampledBAModel,
  h::H,
  options::ROSolverOptions;
  x0::AbstractVector = nls.meta.x0,
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = R2,
  subsolver_options = RegularizedOptimization.ROSolverOptions(ϵa = options.ϵa),
  selected::AbstractVector{<:Integer} = 1:(nls.meta.nvar),
  sample_rate0::Float64 = .05,
  version::Int = 1
) where {H}

  # initializes epoch counting and progression
  epoch_count = 0
  epoch_progress = 0

  # initializes values for adaptive sample rate strategy
  Num_mean = 0
  mobile_mean = 0
  unchange_mm_count = 0
  sample_rates_collec = [.2, .5, .9, .99]
  epoch_limits = [1, 2, 5, 10]
  @assert length(sample_rates_collec) == length(epoch_limits)
  nls.sample_rate = sample_rate0
  ζk = Int(ceil(nls.sample_rate * nls.nobs))
  nls.sample = sort(randperm(nls.nobs)[1:ζk])

  sample_counter = 1
  change_sample_rate = false

  # initialize time stats
  start_time = time()
  elapsed_time = 0.0

  # initialize passed options
  ϵ = options.ϵa
  ϵ_subsolver = subsolver_options.ϵa
  ϵr = options.ϵr
  verbose = options.verbose
  maxIter = options.maxIter
  maxEpoch = maxIter
  maxIter = Int(ceil(maxIter * (nls.nls_meta.nequ / length(nls.sample)))) #computing the sample rate
  maxTime = options.maxTime
  η1 = options.η1
  η2 = options.η2
  η3 = options.η3
  β = options.β
  θ = options.θ
  λ = options.λ
  νcp = options.νcp
  σmin = options.σmin
  σmax = options.σmax
  μmin = options.μmin
  metric = options.metric

  m = nls.nls_meta.nequ

  # store initial values of the subsolver_options fields that will be modified
  ν_subsolver = subsolver_options.ν
  ϵa_subsolver = subsolver_options.ϵa

  if verbose == 0
    ptf = Inf
  elseif verbose == 1
    ptf = round(maxIter / 10)
  elseif verbose == 2
    ptf = round(maxIter / 100)
  else
    ptf = 1
  end

  # initialize parameters
  σk = max(1 / options.ν, σmin)
  μk = max(1 / options.ν , μmin)
  xk = copy(x0)
  hk = h(xk[selected])
  if hk == Inf
    verbose > 0 && @info "SLM: finding initial guess where nonsmooth term is finite"
    prox!(xk, h, x0, one(eltype(x0)))
    hk = h(xk[selected])
    hk < Inf || error("prox computation must be erroneous")
    verbose > 0 && @debug "SLM: found point where h has value" hk
  end
  hk == -Inf && error("nonsmooth term is not proper")
  ψ = shifted(h, xk)

  xkn = similar(xk)

  local ξcp
  local exact_ξcp
  local ξ
  k = 0
  Fobj_hist = zeros(maxIter * 100)
  exact_Fobj_hist = zeros(maxIter * 100)
  Hobj_hist = zeros(maxIter * 100)
  Metric_hist = zeros(maxIter * 100)
  exact_Metric_hist = zeros(maxIter * 100)
  Complex_hist = zeros(Int, maxIter * 100)
  Grad_hist = zeros(Int, maxIter * 100)
  Resid_hist = zeros(Int, maxIter * 100)
  Sample_hist = zeros(maxIter * 100)

  #Historic of time
  TimeHist = []

  if verbose > 0
    #! format: off
    @info @sprintf "%6s %8s %8s %8s %7s %7s %8s %7s %7s %7s %7s %7s %7s %1s %6s" "outer" "inner" "f(x)" "h(x)" "√ξcp/νcp" "√ξ/ν" "ρ" "σ" "μ" "ν" "‖x‖" "‖s‖" "‖Jₖ‖²" "reg" "rate"
    #! format: on
  end

  #creating required objects
  Fk = residual(nls, xk)
  Fkn = similar(Fk)
  exact_Fk = zeros(1:m)

  meta_nls = nls_meta(nls)
  rows = Vector{Int}(undef, meta_nls.nnzj)
  cols = Vector{Int}(undef, meta_nls.nnzj)
  vals = similar(xk, meta_nls.nnzj)
  jac_structure_residual!(nls, rows, cols)
  jac_coord_residual!(nls, nls.meta.x0, vals)

  #sampled Jacobian
  ∇fk = similar(xk)
  JdFk = similar(Fk) # temporary storage
  Jt_Fk = similar(∇fk)

  Jk = jac_op_residual!(nls, rows, cols, vals, JdFk, Jt_Fk)

  fk = dot(Fk, Fk) / 2 #objective estimated without noise
  jtprod_residual!(nls, rows, cols, vals, Fk, ∇fk)

  exact_Jt_Fk = similar(∇fk)

  μmax = opnorm(Jk)
  νcpInv = (1 + θ) * (μmax^2 + μmin)
  νInv = (1 + θ) * (μmax^2 + σk)  # ‖J'J + σₖ I‖ = ‖J‖² + σₖ

  s = zero(xk)
  scp = similar(s)

  optimal = false
  tired = epoch_count ≥ maxEpoch-1 || elapsed_time > maxTime
  #tired = elapsed_time > maxTime

  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk
    Grad_hist[k] = nls.counters.neval_jtprod_residual + nls.counters.neval_jprod_residual
    Resid_hist[k] = nls.counters.neval_residual
    Sample_hist[k] = nls.sample_rate
    if k == 1
      push!(TimeHist, 0.0)
    else
      push!(TimeHist, elapsed_time)
    end

    # model for the Cauchy-Point decrease
    φcp(d) = begin
      jtprod_residual!(nls, rows, cols, vals, Fk, Jt_Fk)
      dot(Fk, Fk) / 2 + dot(Jt_Fk, d)
    end

    #submodel to find scp
    mkcp(d) = φcp(d) + ψ(d) #+ νcpInv * dot(d,d) / 2
    
    #computes the Cauchy step
    νcp = 1 / νcpInv
    ∇fk .*= -νcp
    # take first proximal gradient step s1 and see if current xk is nearly stationary
    # s1 minimizes φ1(s) + ‖s‖² / 2 / ν + ψ(s) ⟺ s1 ∈ prox{νψ}(-ν∇φ1(0)).
    prox!(scp, ψ, ∇fk, νcp)
    #replace!(scp, NaN=>0.0)
    ξcp = fk + hk - mkcp(scp) + max(1, abs(fk + hk)) * 10 * eps()  # TODO: isn't mk(s) returned by subsolver?

    #ξcp > 0 || error("LM: first prox-gradient step should produce a decrease but ξcp = $(ξcp)")
    
    if ξcp ≤ 0
      ξcp = - ξcp
    end

    metric = sqrt(ξcp*νcpInv)
    Metric_hist[k] = metric

    if ξcp ≥ 0 && k == 1
      ϵ_increment = ϵr * metric
      ϵ += ϵ_increment  # make stopping test absolute and relative
      ϵ_subsolver += ϵ_increment
    end

    #=if (metric < ϵ) #checks if the optimal condition is satisfied and if all of the data have been visited
      # the current xk is approximately first-order stationary
      push!(nls.opt_counter, k) #indicates the iteration where the tolerance has been reached by the metric
      if (length(nls.opt_counter) ≥ 3) && (nls.opt_counter[end-2:end] == range(k-2, k)) #if the last 5 iterations are successful
        optimal = true
      end
    end=#

    subsolver_options.ϵa = (length(nls.epoch_counter) ≤ 1 ? 1.0e-1 : max(ϵ_subsolver, min(1.0e-2, metric / 10)))

    #update of σk
    σk = min(max(μk * metric, σmin), σmax)

    # TODO: reuse residual computation
    # model for subsequent prox-gradient iterations

    φ(d) = begin
      jprod_residual!(nls, rows, cols, vals, d, JdFk)
      JdFk .+= Fk
      return dot(JdFk, JdFk) / 2 + σk * dot(d, d) / 2
    end

    ∇φ!(g, d) = begin
      jprod_residual!(nls, rows, cols, vals, d, JdFk)
      JdFk .+= Fk
      jtprod_residual!(nls, rows, cols, vals, JdFk, g)
      g .+= σk * d
      return g
    end

    mk(d) = begin
      jprod_residual!(nls, rows, cols, vals, d, JdFk)
      JdFk .+= Fk
      return dot(JdFk, JdFk) / 2 + σk * dot(d, d) / 2 + ψ(d)
    end
  
    νInv = (1 + θ) * (μmax^2 + σk) # μmax^2 + σk = ||Jmk||² + σk 
    ν = 1 / νInv
    subsolver_options.ν = ν

    @debug "setting inner stopping tolerance to" subsolver_options.optTol
    s, iter, _ = with_logger(subsolver_logger) do
      subsolver(φ, ∇φ!, ψ, subsolver_options, scp)
    end
    # restore initial subsolver_options here so that it is not modified if there is an error
    subsolver_options.ν = ν_subsolver
    subsolver_options.ϵa = ϵa_subsolver

    Complex_hist[k] = iter

    # additionnal condition on step s
    if norm(s) > β * norm(scp)
      println("cauchy step used")
      s .= scp
    end

    xkn .= xk .+ s

    residual!(nls, xkn, Fkn)
    fkn = dot(Fkn, Fkn) / 2
    hkn = h(xkn[selected])
    hkn == -Inf && error("nonsmooth term is not proper")
    mks = mk(s)
    ξ = fk + hk - mks + max(1, abs(hk)) * 10 * eps()

    #=if (ξ ≤ 0 || isnan(ξ))
      error("LM: failed to compute a step: ξ = $ξ")
    end=#

    if ξ ≤ 0
      ξ = - ξ
    end

    Δobj = fk + hk - (fkn + hkn) + max(1, abs(fk + hk)) * 10 * eps()
    #Δobj ≥ 0 || error("Δobj should be positive while Δobj = $Δobj, we should have a decreasing direction but fk + hk - (fkn + hkn) = $(fk + hk - (fkn + hkn))")
    ρk = Δobj / ξ

    #μ_stat = ((η1 ≤ ρk < Inf) && ((metric ≥ η3 / μk))) ? "↘" : "↗"
    μ_stat = ρk < η1 ? "↘" : ((metric ≥ η3 / μk) ? "↗" : "↘")
    #μ_stat = (η2 ≤ ρk < Inf) ? "↘" : (ρk < η1 ? "↗" : "=")

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.4e %7.1e %8.1e %7.1e %7.1e %7.1e %7.1e %7.1e %7.1e %1s %6.2e" k iter fk hk sqrt(ξcp*νcpInv) sqrt(ξ*νInv) ρk σk μk ν norm(xk) norm(s) νInv μ_stat nls.sample_rate
      #! format: off
    end
    
    
    #-- to compute exact quantities --#
    if nls.sample_rate < 1.0
      nls.sample = 1:nls.nobs
      residual!(nls, xk, exact_Fk)
      exact_fk = dot(exact_Fk, exact_Fk) / 2

      exact_φcp(d) = begin
        jtprod_residual!(nls, rows, cols, vals, exact_Fk, exact_Jt_Fk)
        dot(exact_Fk, exact_Fk) / 2 + dot(exact_Jt_Fk, d)
      end

      exact_ξcp = exact_fk + hk - exact_φcp(scp) - ψ(scp) + max(1, abs(fk + hk)) * 10 * eps()
      exact_metric = sqrt(abs(exact_ξcp * νcpInv))

      exact_Fobj_hist[k] = exact_fk
      exact_Metric_hist[k] = exact_metric
    end
    # -- -- #
    

    #updating the indexes of the sampling
    epoch_progress += nls.sample_rate
    if epoch_progress >= 1 #we passed on all the data
      epoch_count += 1
      push!(nls.epoch_counter, k)
      epoch_progress -= 1
    end

    # Version 1: List of predetermined - switch with mobile average #
    if version == 1
      # Change sample rate
      #nls.sample_rate = basic_change_sample_rate(epoch_count)
      if nls.sample_rate < sample_rates_collec[end]
        Num_mean = Int(ceil(1 / nls.sample_rate))
        if k >= Num_mean
          @views mobile_mean = mean(Fobj_hist[(k - Num_mean + 1):k] + Hobj_hist[(k - Num_mean + 1):k])
          if abs(mobile_mean - (fk + hk)) ≤ 1e-1 #if the mean on the Num_mean last iterations is near the current objective value
            nls.sample_rate = sample_rates_collec[sample_counter]
            sample_counter += 1
            change_sample_rate = true
          end
        end
      end
    end

    # Version 2: List of predetermined - switch with arbitrary epochs #
    if version == 2
      if nls.sample_rate < sample_rates_collec[end]
        if epoch_count > epoch_limits[sample_counter]
          nls.sample_rate = sample_rates_collec[sample_counter]
          sample_counter += 1
          change_sample_rate = true
        end
      end
    end

    # Version 3: Adapt sample_size after each iteration #
    if version == 3
      # ζk = Int(ceil(k / (1e8 * min(1, 1 / μk^4))))
      p = .75
      q = .75
      ζk = Int(ceil((log(1 / (1-p)) * max(μk^4, μk^2) + log(1 / (1-q)) * μk^4)))
      nls.sample_rate = min(1.0, (ζk / nls.nobs) * (nls.meta.nvar + 1))
      change_sample_rate = true
    end

    # Version 4: Double sample_size after a fixed number of epochs or a mobile mean stagnation #
    if version == 4
      # Change sample rate
      #nls.sample_rate = basic_change_sample_rate(epoch_count)
      if nls.sample_rate < 1.0
        Num_mean = Int(ceil(1 / nls.sample_rate))
        if k >= Num_mean
          @views mobile_mean = mean(Fobj_hist[(k - Num_mean + 1):k] + Hobj_hist[(k - Num_mean + 1):k])
          if abs(mobile_mean - (fk + hk)) ≤ 1e-1 #if the mean on the Num_mean last iterations is near the current objective value
            nls.sample_rate = min(1.0, 2 * nls.sample_rate)
            change_sample_rate = true
            unchange_mm_count = 0
          else # don't have stagnation
            unchange_mm_count += nls.sample_rate
            if unchange_mm_count ≥ 3 # force to change sample rate after 3 epochs of unchanged sample rate using mobile mean criterion
              nls.sample_rate = min(1.0, 2 * nls.sample_rate)
              change_sample_rate = true
              unchange_mm_count = 0
            end
          end
        end
      end
    end


    #changes sample with new sample rate
    nls.sample = sort(randperm(nls.nobs)[1:Int(ceil(nls.sample_rate * nls.nobs))])

    # mandatory updates whenever the sample_rate chages #
    if change_sample_rate
      #display("Went here for epoch_count = $epoch_count and sample_rate = $(nls.sample_rate)")
      Fk = residual(nls, xk)
      Fkn = similar(Fk)
      JdFk = similar(Fk)
      fk = dot(Fk, Fk) / 2

      Jk = jac_op_residual!(nls, rows, cols, vals, JdFk, Jt_Fk)
      jtprod_residual!(nls, rows, cols, vals, Fk, ∇fk)
      μmax = opnorm(Jk)
      νcpInv = (1 + θ) * (μmax^2 + μmin)

      change_sample_rate = false
    end

    if (η1 ≤ ρk < Inf) #&& (metric ≥ η3 / μk) #successful step
      xk .= xkn

      if (nls.sample_rate < 1.0) && (metric ≥ η3 / μk)  #very successful step
        μk = max(μk / λ, μmin)
      #else
        #μk = λ * μk
      end

      # update functions #FIXME : obligés de refaire appel à residual! après changement du sampling --> on fait des évaluations du résidus en plus qui pourraient peut-être être évitées...
      Fk = residual(nls, xk)
      fk = dot(Fk, Fk) / 2
      hk = hkn

      # update gradient & Hessian
      shift!(ψ, xk)
      Jk = jac_op_residual!(nls, rows, cols, vals, JdFk, Jt_Fk)
      jtprod_residual!(nls, rows, cols, vals, Fk, ∇fk)

      μmax = opnorm(Jk)
      νcpInv = (1 + θ) * (μmax^2 + μmin)

      Complex_hist[k] += 1

    else # (ρk < η1 || ρk == Inf) #|| (metric < η3 / μk) #unsuccessful step
      μk = λ * μk
    end

    tired = epoch_count ≥ maxEpoch-1 || elapsed_time > maxTime
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8s %8.1e %8.1e" k "" fk hk
    elseif optimal
      #! format: off
      @info @sprintf "%6d %8d %8.1e %8.1e %7.4e %7.1e %8s %7.1e %7.1e %7.1e %7.1e %7.1e" k 1 fk hk sqrt(ξcp*νcpInv) sqrt(ξ*νInv) "" σk μk norm(xk) norm(s) νInv
      #! format: on
      @info "SLM: terminating with √ξcp/νcp = $metric"
    end
  end
  status = if optimal
    :first_order
  elseif elapsed_time > maxTime
    :max_time
  elseif tired
    :max_iter
  else
    :exception
  end

  stats = GenericExecutionStats(nls)
  set_status!(stats, status)
  set_solution!(stats, xk)
  set_objective!(stats, fk + hk)
  set_residuals!(stats, zero(eltype(xk)), ξcp ≥ 0 ? sqrt(ξcp * νcpInv) : ξcp)
  set_iter!(stats, k)
  set_time!(stats, elapsed_time)
  set_solver_specific!(stats, :Fhist, Fobj_hist[1:k])
  set_solver_specific!(stats, :ExactFhist, exact_Fobj_hist[1:k])
  set_solver_specific!(stats, :Hhist, Hobj_hist[1:k])
  set_solver_specific!(stats, :NonSmooth, h)
  set_solver_specific!(stats, :SubsolverCounter, Complex_hist[1:k])
  set_solver_specific!(stats, :NLSGradHist, Grad_hist[1:k])
  set_solver_specific!(stats, :ResidHist, Resid_hist[1:k])
  set_solver_specific!(stats, :MetricHist, Metric_hist[1:k])
  set_solver_specific!(stats, :ExactMetricHist, exact_Metric_hist[1:k])
  set_solver_specific!(stats, :TimeHist, TimeHist)
  set_solver_specific!(stats, :SampleRateHist, Sample_hist[1:k])
  return stats
end