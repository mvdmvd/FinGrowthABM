# FinGrowthABM

As an excercise in learning Julia, and creating ABMs, I repoduce Dosi (2003) and Fagiolo (2019) from scratch in Julia, without the original source code.


GrowthBasic.jl is the module for the Dosi(2003) model.

testing.ipynb runs the model and generates the time series.

EmpiricalTestsBasic.ipynb tests the statistical properties to validate the model.

GrowthBanks.jl is the module for the Fagiolo(2003) model.

testing_banks.ipynb runs the model and generates the time series. Note that this model is unfinished: the time series generated does not yet have the desired statistical properties.

GrowthNetwork.jl is my own extension which adds spillovers through a network structure.

testing networks.ipynb tests this model. The result is not very interesting but intuitive: productivity shocks spill over, resulting in a smoother time series.
