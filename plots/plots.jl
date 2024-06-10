include("plot-utils-svm-sto.jl")
include("demo_svm_sto.jl")
include("PLM-plots-svm.jl")
include("PLM-plots-ba.jl")
include("demo_ba_sto.jl")

# Plots for Objective historic, MSE and accuracy #

Random.seed!(seed)

# ---------------- Hyperbolic SVM Models ---------------- #

n_exec = 20
versions = [1, 2, 4]
selected_probs = ["ijcnn1"]

if selected_probs == ["ijcnn1"]
    sample_rate0 = .05
    sample_rates = []
    selected_digits = [(1,7)] # let only one pair of random digits
elseif selected_probs == ["mnist"]
    sample_rate0 = .1
    sample_rates = []
    selected_digits = [(1,7)] # LM crash for (5, 6) and (0, 8)
end

selected_hs = ["l0", "l1", "l1/2"]
abscissas = ["epoch", "CPU time"]
abscissa = abscissas[1]

plot_parameter = ["objective", "metric", "MSE", "accuracy"]
param = plot_parameter[3]

MaxEpochs = 0
MaxTime = 0.0
if abscissa == "epoch"
    MaxEpochs = 100
    MaxTime = 3600.0
elseif abscissa == "CPU time"
    MaxEpochs = 1000
    MaxTime = 10.0
end


plot_Sto_LM_SVM(sample_rates, versions, selected_probs, selected_hs, selected_digits; abscissa = abscissa, n_exec = n_exec, smooth = false, sample_rate0 = sample_rate0, param = param, compare = true, MaxEpochs = MaxEpochs, MaxTime = MaxTime)

# -- Plots for MNIST grey map -- #

Random.seed!(seed)
#=for digits in selected_digits
    demo_svm_sto(;sample_rate = sample_rate0, n_runs = n_exec, digits = digits)
end=#

# ---------------- Bundle Adjustment Models ---------------- #

Random.seed!(seed)

n_exec = 10
versions = [1, 2, 4]
df = problems_df()
filter_df = df[ df.group .== "dubrovnik", :]
sample_rate0 = .1
#name1 = filter_df[1, :name]
name_list = [filter_df[i, :name] for i in 1:3]
selected_hs = ["l1"]
sample_rate0 = .1

plot_parameter = ["objective", "metric", "MSE", "accuracy"]
param = plot_parameter[1]

MaxEpochs = 0
MaxTime = 0.0
if abscissa == "epoch"
    MaxEpochs = 20
    MaxTime = 3600.0
elseif abscissa == "CPU time"
    MaxEpochs = 1000
    MaxTime = 10.0
end

#plot_Sto_LM_BA(sample_rates, versions, name_list, selected_hs; abscissa = abscissa, n_exec = n_exec, smooth = false, sample_rate0 = sample_rate0)
Random.seed!(seed)
#demo_ba_sto(name_list; sample_rate = sample_rate0, n_runs = 3)
