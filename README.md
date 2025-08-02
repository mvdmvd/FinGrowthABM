# FinGrowthABM

Agent-Based Models of Economic Growth, Finance, and Innovation

## Overview

This repository contains Julia implementations of an agent-based model in economic growth theory reproduced from scratch as an exercise in learning Julia and agent-based modeling (ABM).

## Implementations

### 1. Basic Growth Model (Dosi et al., 2003)
**File:** `GrowthBasic.jl`

Simulates economic growth through random exploration. Firms act as agents that randomly explore a ''technology space''. As other firms immitate succesful innovators, technology diffuses through the economy. This generates a ''GDP'' time series that is statistically indistinguishable from the real data.
### 2. Growth with Banking (Fagiolo et al., 2019) 
**File:** `GrowthBanks.jl`

An extension of the basic model that incorporates banks and credit. This model analyzes the effect of credit constraints on economic growth. Note that it is unfinished as the produced time-series are not as desired.


### 3. Network Growth Model (Original Extension)
**File:** `GrowthNetwork.jl`

I extend the basic model by connecting the firms through a Holme-Kim network topology. The point is to incorporate productivity spillovers through supply chains. The result is that the time-series becomes smoother, as firms immediately benefit from succesfully innovating neighbours.

## Repository 

```
├── GrowthBasic.jl           # Core Dosi (2003) model implementation
├── GrowthBanks.jl           # Banking extension (Fagiolo 2019)
├── GrowthNetwork.jl         # Network spillovers extension
├── testing.ipynb           # Basic model simulation and analysis
├── testing_banks.ipynb     # Banking model experiments
├── testing networks.ipynb  # Network model experiments  
├── EmpiricalTestsBasic.ipynb # Statistical validation tests
├── Original models/         # Reference papers (PDF)
└── README.md
```

## Dependencies

- **Julia** (≥ 1.6)
- **Agents.jl** - Agent-based modeling framework
- **Distributions.jl** - Statistical distributions
- **Random.jl** - Random number generation
- **Graphs.jl** - Network structures (for network model)

## Usage

### Running the Basic Model

```julia
using Pkg
Pkg.activate(".")

include("GrowthBasic.jl")
using .GrowthBasic

# Initialize and run simulation
model = initialize_econ(n_firms=100, dims=(50,50))
run!(model, agent_step!, model_step!, 1000)
```

### Notebooks

Open the Jupyter notebooks for interactive exploration:
- `testing.ipynb` - Basic model experiments
- `EmpiricalTestsBasic.ipynb` - Statistical analysis and validation
- `testing_banks.ipynb` - Banking model (experimental)
- `testing networks.ipynb` - Network spillovers analysis


## References

- Dosi, G., Fagiolo, G., & Roventini, A. (2003). *Innovation, finance, and economic growth: An agent-based approach*
- Fagiolo, G., Guerini, M., Lamperti, F., Moneta, A., & Roventini, A. (2019). *Validation of agent-based models in economics and finance*


## License

MIT License - See LICENSE file for details
