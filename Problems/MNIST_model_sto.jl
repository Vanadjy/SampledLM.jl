function tan_data_train(args...)
    #load data
    A, b = MLDatasets.MNIST(split = :train)[:]
    A, b = generate_data(A, b, args...)
    return A, b
end

function tan_data_test(args...)
    A, b = MLDatasets.MNIST(split = :test)[:]
    A, b = generate_data(A, b, args...)
    return A, b
end

function generate_data(A, b, digits::Tuple{Int, Int} = (1, 7), switch::Bool = false)
    length(digits) == 2 || error("please supply two digits only")
    digits[1] != digits[2] || error("please supply two different digits")
    all(0 .≤ digits .≤ 9) || error("please supply digits from 0 to 9")
    ind = findall(x -> x ∈ digits, b)
    #reshape to matrix
    A = reshape(A, size(A, 1) * size(A, 2), size(A, 3)) ./ 255
  
    #get 0s and 1s
    b = float.(b[ind])
    b[b .== digits[2]] .= -1
    A = convert(Array{Float64, 2}, A[:, ind])
    if switch
      p = randperm(length(b))[1:Int(floor(length(b) / 3))]
      b = b[p]
      A = A[:, p]
    end
    return A, b
end

function MNIST_test_model_sto(sample_rate; digits::Tuple{Int, Int} = (1, 7), switch::Bool = false)
    A, b = tan_data_test(digits, switch)
    return svm_model_sto(A, b; sample_rate = sample_rate)
end

function MNIST_train_model_sto(sample_rate; digits::Tuple{Int, Int} = (1, 7), switch::Bool = false)
    A, b = tan_data_train(digits, switch)
    return svm_model_sto(A, b; sample_rate = sample_rate)
end