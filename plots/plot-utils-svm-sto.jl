import PGFPlots

function plot_svm(outstruct, sol, name="sto-lm")
    Comp_pg = outstruct.solver_specific[:SubsolverCounter]
    objdec = outstruct.solver_specific[:Fhist] + outstruct.solver_specific[:Hhist]
    x = outstruct.solution
    a = PGFPlots.Axis(
        [
            PGFPlots.Plots.MatrixPlot(reshape(x, 28, 28); #filename="svm-$(name).tikz",
            colormap = PGFPlots.ColorMaps.GrayMap(), zmin = -700, zmax = 700)#, legendentry="computed"),
            # PGFPlots.Plots.Linear(1:length(sol), sol, mark="none", legendentry="exact"),
        ],
        # xlabel="index",
        # ylabel="parameter",
        # legendStyle="at={(1.0,1.0)}, anchor=north east, draw=none, font=\\scriptsize",
    )
    PGFPlots.save("svm-$(name).tikz", a)

    #=b = PGFPlots.Axis(
        PGFPlots.Plots.Linear(1:length(Comp_pg), Comp_pg, mark="none"),
        xlabel="outer iterations",
        ylabel="inner iterations",
        ymode="log",
    )
    PGFPlots.save("svm-inner-outer-$(name).tikz", b)

    c = PGFPlots.Axis(
        PGFPlots.Plots.Linear(1:length(objdec), objdec, mark="none"),
        xlabel="\$ k^{th}\$  \$ \\nabla f \$ Call",
        ylabel="Objective Value",
        ymode="log",
    )
    PGFPlots.save("svm-objdec-$(name).tikz", c)=#
    return objdec
end

function plot_svm_sto(sol, name="sto-lm")
    #Comp_pg = outstruct.solver_specific[:SubsolverCounter]
    #objdec = outstruct.solver_specific[:Fhist] + outstruct.solver_specific[:Hhist]
    x = sol
    a = PGFPlots.Axis(
        [
            PGFPlots.Plots.MatrixPlot(reshape(x, 28, 28); #filename="svm-$(name).tikz",
            colormap = PGFPlots.ColorMaps.GrayMap(), zmin = -700, zmax = 700)#, legendentry="computed"),
            # PGFPlots.Plots.Linear(1:length(sol), sol, mark="none", legendentry="exact"),
        ],
        # xlabel="index",
        # ylabel="parameter",
        # legendStyle="at={(1.0,1.0)}, anchor=north east, draw=none, font=\\scriptsize",
    )
    PGFPlots.save("svm-$(name).tikz", a)
    return a
end