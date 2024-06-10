function lrcomp_data(m::Int, n::Int; T::DataType = Float64)
    A = Array(rand(T, (m, n)))
    A
  end
  
  function lrcomp_model(m::Int, n::Int; T::DataType = Float64, sample_rate = 1.0)
    A = lrcomp_data(m, n, T = T)

    #initializes sampling parameters
    sample = sort(randperm(size(A, 1) * size(A, 2))[1:Int(sample_rate * size(A, 1) * size(A, 2))])
    data_mem = copy(sample)

    r = vec(similar(A))[sample]

    function resid!(r, x; sample = sample)
      for i in 1:length(sample)
        r[i] = x[sample[i]] - A[sample[i]]
      end
      r
    end
  
    function jprod_resid!(Jv, x, v; sample = sample)
      display(length(v))
      display(length(sample))
      Jv[1:length(sample)] .= v
      Jv
    end
  
    function obj(x)
      resid!(r, x)
      dot(r, r) / 2
    end
  
    grad!(r, x) = resid!(r, x)
    nlsmodel_kwargs = Dict{Symbol, Any}(:name => "LRCOMP-LS")
  
    x0 = rand(T, m * n)
    FirstOrderModel(obj, grad!, x0, name = "LRCOMP"),
    SampledNLSModel(resid!, jprod_resid!, jprod_resid!, m * n, x0, name = "LRCOMP-LS", sample, data_mem, sample_rate; nlsmodel_kwargs...),
    vec(A)
  end