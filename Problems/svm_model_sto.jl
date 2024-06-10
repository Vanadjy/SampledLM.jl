function svm_model_sto(A, b; sample_rate::AbstractFloat = 1.0)
  Ahat = Diagonal(b) * A' #dimensions : m × n

  #initializes sampling parameters
  sample = sort(randperm(size(Ahat,1))[1:Int(ceil(sample_rate * size(Ahat,1)))])
  data_mem = copy(sample)
  r = similar(b[1:length(sample)])
  tmp = similar(r)

  function resid!(r, x; sample = sample)
    mul!(r, Ahat[sample, :], x)
    r .= 1 .- tanh.(r)
    r
  end

  function jacv!(Jv, x, v; sample = sample)
    r = similar(b[1:length(sample)])
    mul!(r, Ahat[sample, :], x)
    mul!(Jv, Ahat[sample, :], v)
    Jv .= -((sech.(r)) .^ 2) .* Jv
  end

  function jactv!(Jtv, x, v; sample = sample)
    r = similar(b[1:length(sample)])
    tmp = similar(r)
    mul!(r, Ahat[sample, :], x)
    tmp .= sech.(r) .^ 2
    tmp .*= v
    tmp .*= -1
    mul!(Jtv, Ahat[sample, :]', tmp)
  end

  function obj(x)
    resid!(r, x)
    dot(r, r) / 2
  end
  function grad!(g, x; sample = sample)
    mul!(r, Ahat[sample, :], x)
    tmp .= (sech.(r)) .^ 2
    tmp .*= (1 .- tanh.(r))
    tmp .*= -1
    mul!(g, Ahat[sample, :]', tmp)
    g
  end

  FirstOrderModel(obj, grad!, ones(size(A, 1)), name = "Nonlinear-SVM"),
  SampledNLSModel(resid!, jacv!, jactv!, length(b), ones(size(A, 1)), sample, data_mem, sample_rate),
  b
end