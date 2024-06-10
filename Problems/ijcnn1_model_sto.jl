using SparseArrays
using Plots

function ijcnn1_load_data(n::Int = 49990, d::Int = 22)
    A = zeros(n,d)
    y = zeros(n)
    f = open(raw"C:\Users\valen\Desktop\Polytechnique_Montreal\_maitrise\data_files\ijcnn1\ijcnn1.txt")
    lines = readlines(f)
    i = 1
    while i ≤ n
        dummy = split(lines[i], ' ')
        y[i] = parse(Float64, dummy[1])
        for j in 2:(length(dummy))
            loc_val = split(dummy[j],':')
            if length(loc_val) < 2
                display(i)
                display(loc_val)
            end
            A[i, parse(Int, loc_val[1])] = parse(Float64, loc_val[2])
        end
    i += 1
    end
    close(f)
    A, y
end

function ijcnn1_model_sto(sample_rate)
    A, b = ijcnn1_load_data()
    return svm_model_sto(A', b; sample_rate = sample_rate)
end