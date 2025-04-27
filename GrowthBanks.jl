module GrowthBanks
using Agents, Agents.Pathfinding, Random, Distributions

export Bank, Firm, initialize_econ, firmbank_step!

## In this file I write the agent classes, their step functions, and the model init function ##

##############################################################
### Agents ###################################################
##############################################################
@agent struct Bank(GridAgent{2})
    state::String # Placeholder, the package wants every agent to be on the grid

    liquidity::Float64
    equity::Float64
    L::Float64

    firms::Vector{AbstractAgent}
    dividend::Float64
end

@agent struct Firm(GridAgent{2})
    state::String # mi, im or ex
    Q::Float64 # productivity
    S::Float64 # savings
    D::Float64 # debt
    bank::Union{Bank,Nothing} # bank assigned to
    share::Float64
    wants_loan::Vector{Any}
    mem::Tuple{Tuple{Int,Int},Float64} # memory of last island mined
    target::Tuple{Int,Int} # target island for imitator
end


######################################################################
### Step Functions ###################################################
######################################################################
function firmbank_step!(firm::Firm, model)
    # take params from model
    α = model.α
    ϵ = model.ϵ
    λ = model.λ
    ϕ = model.ϕ
    ρ = model.ρ
    c = model.c
    r = model.r
    γ₁ = model.γ₁
    ξ = model.ξ
    π_isl = model.π_isl
    middle = model.middle # agents start in the middle of the space
    pathfinder = model.pathfinder  # pathfinder algo finds shortest path to target for imitator

    pos = firm.pos
    _, old_Q = firm.mem
    ############# Behaviour for Explorers ##################################
    ############# 1. Cost #############   
    if firm.state == "ex"
        cost = c * old_Q # cost of exploration
        firm.S -= cost
        bank = firm.bank
        liquidity = min(cost, bank.liquidity)
        bank.liquidity -= liquidity
        bank.equity -= cost - liquidity
        firm.S = max(firm.S, 0.0)
        bank.liquidity = max(bank.liquidity, 0.0)
        bank.equity = max(bank.equity, 0.0)
        ############# 2. Movement #############
        randomwalk!(firm, model, 1) #Take one random-walk step
        pos = firm.pos
        if model.island_dummy[pos...] == true #If they find an island
            firm.state = "mi" # Become Miner
            if model.discovered[pos...] == false # Set new productivity if it was undiscovered
                model.discovered[pos...] = true
                W = rand(model.rng, Poisson(λ)) # Get lucky sometimes
                ω = rand(model.rng, Normal(0, 1)) # Noise
                x = abs(pos[1] - middle) # Distance from origin
                y = abs(pos[2] - middle) # Take out middle to adjust for starting in centre of matrix
                model.productivity[pos...] = (1 + W) * (x + y + ϕ * old_Q + ω)
            end
            m_pos = length(agents_in_position(firm, model))
            firm.Q = model.productivity[pos...] * (m_pos)^(α - 1) # Set new firm productivity
        end

        ############ Behaviour for Immitators ##############################################
        ############# 1. Cost #############
    elseif firm.state == "im"
        cost = c * old_Q # cost of exploration
        firm.S -= cost
        bank = firm.bank
        liquidity = min(cost, bank.liquidity)
        bank.liquidity -= liquidity
        bank.equity -= cost - liquidity
        firm.S = max(firm.S, 0.0)
        bank.liquidity = max(bank.liquidity, 0.0)
        bank.equity = max(bank.equity, 0.0)
        ############# 2. Movement #############
        move_along_route!(firm, model, pathfinder) # step along the shortest path to destination
        pos = firm.pos
        if pos == firm.target
            firm.state = "mi"
            m_pos = length(agents_in_position(firm, model))
            firm.Q = model.productivity[pos...] * (m_pos)^(α - 1)
        end

        ############# Behaviour for Miners ####################################################
        ############# 1. Cost, Repayment, Saving #############
    elseif firm.state == "mi"
        m_pos = length(agents_in_position(firm, model))
        firm.Q = model.productivity[pos...] * (m_pos)^(α - 1) # update productivity for new miners
        Q = firm.Q
        bank = firm.bank
        pos = firm.pos
        if firm.wants_loan[1] == "granted" # transform if loan was requested previous period
            firm.wants_loan[1] = false
            firm.state = firm.wants_loan[2]
            firm.mem = (pos, Q) # set memory
            firm.Q = 0.0 # no production during innovation
        else
            firm.D = max(firm.D, 0.0)
            repay = min(firm.D, Q / (1 + r)) # repay with interest
            firm.D -= repay
            bank.L -= repay
            bank.dividend += repay * r * ξ # share of profits becomes dividend
            bank.liquidity += repay * (1 + r * (1 - ξ)) # the rest is for the bank
            firm.D = max(firm.D, 0.0)
            bank.L = max(bank.L, 0.0)

            S = (1 - c) * (Q - repay * (1 + r)) # invest and save
            bank.equity += γ₁ * S
            firm.share += γ₁ * S # share invested in bank equity
            bank.liquidity += (1 - γ₁) * S
            firm.S += (1 - γ₁) * S # rest is deposited

            ############# 2. Explorer transition #############
            if rand(model.rng) ≤ ϵ && firm.D ≤ 1e-5 # with prob ϵ and if no debt, transform. (small positive const for numerical stability)
                C = Q / π_isl # expected cost of exploration
                if firm.S ≥ C # if firm has the funds, become explorer
                    firm.state = "ex"
                    firm.mem = (firm.pos, Q)
                    firm.Q = 0.0
                else
                    firm.wants_loan = [true, "ex", (C - firm.S)]
                end
            else
                ############# 3. Receive signals and become immitator #############
                all_mi = filter(agent -> agent.state == "mi", collect(allagents(model))) # list of all miners
                other_islands = filter(agent -> agent.pos != pos, all_mi) # filter out those on same island
                signals = Dict((middle, middle) => model.productivity[pos...]) # dictionary of all miners on other miners and their productivities
                m = length(all_mi)
                for miner in other_islands # compute probability that a given signal is received
                    firms = collect(agents_in_position(miner.pos, model))
                    mⱼ = count(firm -> firm.state == "mi", firms) # number of other miners on island where signal from
                    distance = manhattan_distance(firm, miner, model)
                    wⱼ = (mⱼ / m) * exp(-ρ * distance) # probability of signal success
                    rand(model.rng, Binomial(1, wⱼ)) == 1 ? (signals[miner.pos...] = model.productivity[miner.pos...]) : nothing # if succesfull, add to dict
                end
                if !isempty(signals)
                    signal_prod, signal_pos = findmax(signals) # find the highest productivity signal
                    current_prod = model.productivity[pos...]
                    if signal_prod > current_prod && firm.D ≤ 1e-5 # If its better than known and no debt
                        dist = manhattan_distance(firm.pos, signal_pos, model)
                        C = c * Q * dist
                        if firm.S ≥ C
                            firm.state = "im" # Turn into immitator
                            firm.mem = (firm.pos, Q)
                            firm.Q = 0.0 # No output
                            firm.target = signal_pos #Target island
                            plan_route!(firm, signal_pos, pathfinder)
                        else
                            firm.wants_loan = [true, "im", (C - firm.S)]
                        end
                    end
                end
            end
        end
    end
end

function firmbank_step!(bank::Bank, model)
    Χ = model.Χ
    α = model.α
    ζ = model.ζ
    γ₂ = model.γ₂
    c = model.c

    ############# 1. Select firms and provide loans #############
    applicants = filter(firm -> firm.wants_loan[1] == true, bank.firms)
    sorted_applicants = sort(applicants, by=firm -> firm.wants_loan[3])
    credit_supply = max((bank.equity / Χ) - bank.L, 0)
    total_requested = 0.0
    for firm in sorted_applicants
        amount = firm.wants_loan[3]
        if total_requested + amount ≥ credit_supply
        end
        if total_requested + amount ≤ credit_supply
            total_requested += amount
            firm.D += amount
            bank.L += amount
            firm.S += amount
            firm.wants_loan[1] = "granted"
            firm.wants_loan[3] = 0.0

        end
    end

    ############# 2. Process defaults and pay dividends #############
    defaulted_shares = 0.0
    for firm in bank.firms
        if firm.S ≤ 0.0 #if firm default
            bank.equity -= max(firm.D, 0.0)
            defaulted_shares += firm.share
            firm.share = 0.0
            firm.S = 0.0
            firm.D = 0.0
            firm.state = "mi"
            old_pos, _ = firm.mem
            move_agent!(firm, old_pos, model)
            m_pos = length(agents_in_position(firm, model))
            firm.Q = model.productivity[old_pos...] * (m_pos)^(α - 1)
        end
    end
    for firm in bank.firms
        equity_share = firm.share / bank.equity
        firm.S += (bank.dividend) * equity_share
        firm.share += (defaulted_shares) * equity_share
    end
    bank.dividend = 0.0

    ############# 3. Process bankrupcy #############
    if bank.equity ≤ 1e-5
        bank.equity = 0.0
        tot_deposits = 0.0
        tot_shares = 0.0
        for firm in bank.firms
            tot_shares += firm.share
            net_pos = min(firm.S - firm.D, 0.0)
            tot_deposits += max(0.0, net_pos)
        end
        if bank.liquidity ≥ tot_deposits # pay back deposits fully if liquitity allows
            for firm in bank.firms
                net_pos = min(firm.S - firm.D, 0.0)
                firm.S = max(net_pos, 0.0)
                bank.liquidity -= net_pos
            end
            for firm in bank.firms # then distribute remainder according to equity share
                firm.S += bank.liquidity * (firm.share / tot_shares)
            end
        else # otherwise pay back deposits proportional to net position
            for firm in bank.firms
                firm.S = bank.liquidity * (min(firm.S - firm.D, 0.0) / tot_deposits)
            end
        end
        bank.liquidity = 0.0
        for firm in bank.firms # make sure savings are deposited
            bank.liquidity += firm.S
            firm.D = 0.0
        end

        # recapitalisation
        new_fund = 0.0
        for firm in bank.firms
            new_share = firm.Q * (1 - c) * γ₂
            firm.S -= new_share
            firm.share = new_share
            new_fund += new_share
        end
        bank.equity = new_fund * ζ
    end




end

####################################################################################################
### Initialization #################################################################################
####################################################################################################
function initialize_econ(;
    n_firms=100,
    n_banks=5,
    dim=1001, # Always make uneven!
    π_isl=0.1,
    α=1.5,
    ϵ=0.1,
    λ=1,
    ϕ=0.5,
    ρ=0.1,
    c=0.7,
    Χ=0.1,
    r=0.1,
    γ₁=0.01,
    γ₂=0.5,
    ζ=0.1,
    ξ=0.15,
    seed=42)

    middle = dim ÷ 2 + 1 # Selects the middle of the uneven dim, so that we spawn all the agents in the center of the space
    rng = MersenneTwister(seed)

    ############# 1. Set model properties #############
    tech_space = GridSpace((dim, dim); periodic=false, metric=:manhattan)#Manhattan as per the paper, periodic turns it into a donut, we just make it larger than the agents can reach within T
    pathfinder = AStar(tech_space; diagonal_movement=false) # Specify which algo finds the shortest path for the immitator agents

    properties = (
        island_dummy=falses(dim, dim), # We will set some to True based on prob of island
        discovered=falses(dim, dim),
        productivity=ones(dim, dim), # New prods are set once the islands are discovered
        π_isl=π_isl,
        α=α,
        ϵ=ϵ,
        λ=λ,
        ϕ=ϕ,
        ρ=ρ,
        c=c,
        Χ=Χ,
        r=r,
        γ₁=γ₁,
        γ₂=γ₂,
        ζ=ζ,
        ξ=ξ,
        middle=middle,
        pathfinder=pathfinder,
        rng=rng
    )

    ############# 2. Initialize model #############
    model = StandardABM(Union{Bank,Firm}, tech_space; properties=properties, (agent_step!)=firmbank_step!, rng=rng,
        scheduler=Schedulers.fastest,
        warn=true)

    ############# 3. Populate with firms then banks #############
    firms = []
    #Adding agents
    for _ in 1:n_firms
        state = "mi"
        Q = 0.0
        S = 0.0
        D = 0.0
        bank = nothing
        wants_loan = [false, "placeholder"]
        share = 0.0
        pos = (middle, middle)
        mem = (pos, 0.0)
        firm = add_agent!(Firm, model; state=state, Q=Q, S=S, D=D, share=share, bank=bank, wants_loan=wants_loan, mem=mem, pos=pos, target=pos)
        push!(firms, firm)
    end

    #Assinging firms to banks
    firm_groups = [Vector{AbstractAgent}() for _ in 1:n_banks]
    for (i, firm) in enumerate(firms)
        push!(firm_groups[i%n_banks+1], firm)
    end

    #Adding banks
    for i in 1:n_banks
        add_agent!(Bank, model;
            state="bank",
            liquidity=0.0,
            L=0.0,
            equity=0.0,
            dividend=0.0,
            firms=firm_groups[i],
            pos=(middle, middle))
    end

    #Assign banks to firms
    for bank in allagents(model)
        if isa(bank, Bank)
            for firm in bank.firms
                firm.bank = bank
            end
        end
    end

    ############# 4. Generate islands
    for p in positions(model)
        model.island_dummy[p...] = rand(model.rng) ≤ π_isl
        model.discovered[middle, middle] = true
    end

    ############# 5. Assign initial Q
    for firm in allagents(model)
        if isa(firm, Firm)
            firm.Q = (n_firms)^(α - 1)
        end
    end

    # initial capitalisation
    for bank in allagents(model)
        if isa(bank, Bank)
            for firm in bank.firms
                share = firm.Q * (1 - c) * γ₂
                bank.equity += share
                firm.share += share
            end
        end
    end
    return model
end

end
