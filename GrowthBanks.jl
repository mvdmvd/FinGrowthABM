module GrowthBanks
using Agents, Agents.Pathfinding, Random, Distributions

export Bank, Firm, initialize_econ, firmbank_step!

# TODO
# Savings have to be deposits!!!! or this thing doesnt work!!!!!!
# add a new attribute deposits and whenever firm.E increases then bank.deposits increases!
# liabilities are first settled with deposits and then with capital
## In this file I write the agent classes, their step functions, and the model init function ##

##############################################################
### Agents ###################################################
##############################################################
@agent struct Bank(GridAgent{2})
    state::String # Placeholder, the package wants every agent to have a pos

    capital::Float64
    loans::Float64

    liabilities::Float64
    equity::Float64

    firms::Vector{AbstractAgent}
    profits::Float64
end

@agent struct Firm(GridAgent{2})
    state::String # mi, im or ex
    Q::Float64 # productivity
    E::Float64 # savings
    D::Float64 # debt
    B::Float64 # bank money
    bank::Union{Bank,Nothing} # bank assigned to
    wants_loan::Vector{Any}
    amount::Float64 # amount of loan wanted
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
    middle = model.middle # agents start in the middle of the space
    pathfinder = model.pathfinder  # pathfinder algo finds shortest path to target for imitator


    _, old_Q = firm.mem
    ############# Behaviour for Explorers ##################################
    ############# 1. Pay exploration cost #############   
    if firm.state == "ex"
        # First cost pay by bank, then by savings           
        cost = c * old_Q # cost of exploration
        if firm.B > 0.0
            bybank = min(firm.B, cost)
            firm.B -= bybank
            firm.E -= (cost - bybank)
            bank = firm.bank
            bank.capital -= bybank
            bank.liabilities -= bybank
        else
            firm.E -= cost
        end

        ############# 3. Exploration step #############
        randomwalk!(firm, model, 1) #Take one random-walk step
        position = firm.pos
        if model.island_dummy[position...] == true #If they find an island
            firm.state = "mi" # Become Miner
            if model.disc_dummy[position...] == false ## If it was undiscovered
                model.disc_dummy[position...] = true
                #Get vars for the new island productivity
                W = rand(model.rng, Poisson(λ)) # Get lucky sometimes
                ω = rand(model.rng, Normal(0, 1)) # Noise
                x = abs(position[1] - middle) # Distances from "origin"
                y = abs(position[2] - middle) # Take out middle to adjust for starting in centre of matrix
                # New island productivity coefficient
                model.productivity[position...] = (1 + W) * (x + y + ϕ * old_Q + ω)
                # Firm productivity updated  
            end
            m_pos = length(agents_in_position(firm, model))
            firm.Q = model.productivity[position...] * (m_pos)^(α - 1) #set new firm productivity
        end

        ############ Behaviour for Immitators ##############################################
        ############# 1. Immitation cost #############
    elseif firm.state == "im"
        # First by bank, then by savings
        cost = c * old_Q # cost of immitation
        if firm.B > 0.0
            bybank = min(firm.B, cost)
            firm.B -= bybank
            firm.E -= (cost - bybank)
            bank = firm.bank
            bank.capital -= bybank
            bank.liabilities -= bybank
        else
            firm.E -= cost
        end

        ############# 3. Check if destination reached, become miner if so #############
        move_along_route!(firm, model, pathfinder)
        position = firm.pos
        if position == firm.target
            firm.state = "mi"
            m_pos = length(agents_in_position(firm, model))
            firm.Q = model.productivity[position...] * (m_pos)^(α - 1)
        end

        ############# Behaviour for Miners ####################################################
        ############# 1. Pay off loans, then save #############
    elseif firm.state == "mi"
        Q = firm.Q
        bank = firm.bank
        position = firm.pos
        if firm.wants_loan[1] == "granted"
            firm.state = firm.wants_loan[2]
            firm.mem = (firm.pos, Q)
            firm.Q = 0.0
            firm.wants_loan[1] = false
        else
            cost = c * Q # consume
            if firm.B > 0.0
                bybank = min(firm.B, cost)
                firm.B -= bybank
                firm.E -= (cost - bybank)
                bank = firm.bank
                bank.capital -= bybank
                bank.liabilities -= bybank
            else
                firm.E -= cost
            end
            # pay off debt then save
            S = Q * (1 - c)
            if firm.D > 0 # if firm has debt, repay
                repayment = min(firm.D, S / (1 + r))
                firm.D -= repayment # discount to obtain the principal
                bank.loans -= repayment # outstanding loan turned into capital
                bank.capital += repayment
                bank.equity += (repayment * r * (1 - ξ)) # profits become equity
                bank.capital += (repayment * r * (1 - ξ))
                bank.profits += (repayment * r * ξ)
                save = max(S - (repayment * (1 + r)), 0)
                bank.capital += γ₁ * save
                bank.equity += γ₁ * save
                firm.E = (1 - γ₁) * save # save remainder
            else # save and deposit
                bank.capital += γ₁ * S
                bank.equity += γ₁ * S
                firm.E = (1 - γ₁) * S
            end
            ############# 2. Explorer transition, if funds, else wants_loan #############
            if rand(model.rng) ≤ ϵ && firm.D ≤ 0.1 # with prob ϵ
                m_pos = length(agents_in_position(firm, model))
                C = Q / model.π_isl #expected cost of exploration
                if (firm.E + firm.B) ≥ C
                    firm.state = "ex"
                    firm.mem = (firm.pos, Q)
                    firm.Q = 0.0
                else
                    firm.wants_loan = [true, "ex"]
                    firm.amount = max(C - (firm.E + firm.B), 0)
                end
                ############# 3. Receive signals and become immitator #############
            else
                mi_agents = filter(agent -> agent.state == "mi", collect(allagents(model))) # List of all miners
                other_islands = filter(agent -> agent.pos != position, mi_agents) # List of miners on other islands
                signals_received = Dict((middle, middle) => model.productivity[position...]) #
                m = length(mi_agents) #Total miners
                for miner in other_islands
                    # All miners on island where signal from
                    firms = collect(agents_in_position(miner.pos, model))
                    mⱼ = count(firm -> firm.state == "mi", firms)
                    distance = manhattan_distance(firm, miner, model)
                    wⱼ = (mⱼ / m) * exp(-ρ * distance)
                    received = rand(model.rng, Binomial(1, wⱼ)) #Chance that signal is received
                    if received == 1 # if the signal was received
                        signals_received[miner.pos...] = model.productivity[miner.pos...]
                    end
                end
                if !isempty(signals_received)
                    new_coef, new_pos = findmax(signals_received) # Select the best signal
                    old_coef = model.productivity[position...]
                    if new_coef > old_coef && firm.D ≤ 0.1 # If its better than known and no debt
                        m_pos = length(agents_in_position(firm, model))
                        distance = manhattan_distance(firm.pos, new_pos, model)
                        C = c * Q * distance
                        if (firm.E + firm.B) ≥ C
                            firm.state = "im" # Turn into immitator
                            firm.mem = (firm.pos, firm.Q)
                            firm.Q = 0.0 # No output
                            firm.target = new_pos #Target island
                            plan_route!(firm, new_pos, pathfinder)
                        else
                            firm.wants_loan = [true, "im"]
                            firm.amount = max(C - (firm.E + firm.B), 0)
                        end
                    end
                end
            end
        end
    end
end

function firmbank_step!(bank::AbstractAgent, model)
    Χ = model.Χ
    α = model.α
    ζ = model.ζ
    γ₂ = model.γ₂


    div = bank.profits / length(bank.firms)
    ############# 1. Process defaults and pay dividends #############
    for firm in bank.firms
        if firm.E + firm.B ≤ 0.0 #firm default
            bank.loans -= firm.D
            bank.equity -= firm.D
            old_pos, _ = firm.mem
            firm.state = "mi"
            move_agent!(firm, old_pos, model)
            m_pos = length(agents_in_position(firm, model))
            firm.Q = model.productivity[old_pos...] * (m_pos)^(α - 1)
            firm.E = 0.0
            firm.B = 0.0
            firm.D = 0.0
        end
        firm.E += div
    end
    bank.profits = 0.0

    ############# 2. Process bankrupcy #############
    if bank.equity ≤ 0.0
        for firm in bank.firms
            firm.E += bank.capital \ length(bank.firms)
        end
        recap = 0.0
        for firm in bank.firms
            share = firm.E * γ₂
            firm.E -= share
            recap += share
        end
        bank.capital = recap * ζ

        bank.equity = bank.capital + bank.loans - bank.liabilities

    end

    ############# 3. Select firms and provide loans #############
    firms_looking = filter(firm -> firm.wants_loan[1] == true, bank.firms)
    sorted_firms_looking = sort(firms_looking, by=firm -> firm.amount)
    c_supp = max((bank.equity / Χ) - bank.loans, 0)

    total_req = 0.0
    firms_supplied = []

    for firm in sorted_firms_looking
        if total_req + firm.amount ≤ c_supp
            total_req += firm.amount
            push!(firms_supplied, firm)
        end
    end

    for firm in firms_supplied
        amount = firm.amount
        firm.D += amount
        firm.B += amount
        firm.amount = 0.0
        firm.wants_loan[1] = "granted"
        bank.loans += amount
        bank.liabilities += amount
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
        disc_dummy=falses(dim, dim),
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
        E = 0.0
        D = 0.0
        B = 0.0
        bank = nothing
        wants_loan = [false, "placeholder"]
        amount = 0.0
        pos = (middle, middle)
        mem = (pos, 0.0)
        firm = add_agent!(Firm, model; state=state, Q=Q, E=E, D=D, B=B, bank=bank, wants_loan=wants_loan, amount=amount, mem=mem, pos=pos, target=pos)
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
            capital=0.0,
            loans=0.0,
            pos=(middle, middle),
            liabilities=0.0,
            equity=0.0,
            firms=firm_groups[i],
            profits=0.0)
    end

    for bank in allagents(model) #Assign banks to firms
        if isa(bank, Bank)
            for firm in bank.firms
                firm.bank = bank
            end
        end
    end


    ############# 4. Generate islands
    for p in positions(model)
        model.island_dummy[p...] = rand(model.rng) ≤ π_isl
        model.disc_dummy[middle, middle] = true
    end

    ############# 5. Assign initial Q
    for firm in allagents(model)
        if isa(firm, Firm)
            firm.Q = (n_firms)^(α - 1)
        end
    end

    for bank in allagents(model) #Assign banks to firms
        if isa(bank, Bank)
            cap = 0.0
            for firm in bank.firms
                cap_i = firm.Q * γ₂
                cap += cap_i
            end
            bank.equity = cap
            bank.capital = cap
        end
    end

    return model
end

end
