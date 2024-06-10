include("bpdn_model_sampled_sto.jl")
include("group_lasso_model_sto.jl")
include("matrand_model_sto.jl")
include("lrcomp_model_sto.jl")

# hyperbolic SVM problems
include("svm_model_sto.jl")
include("ijcnn1_model_sto.jl")
include("a9a_model_sto.jl")
include("MNIST_model_sto.jl")

# Bundle Adjustment problems
include("BA_jac_by_hand.jl")
include("ba_model_sto.jl")
#include("demo_svm.jl")
#include("plot-utils-svm.jl")
#include("bpdn_model_sampled_prob.jl")