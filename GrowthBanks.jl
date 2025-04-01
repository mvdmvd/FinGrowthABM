module GrowthBanks
using Agents, Agents.Pathfinding, Random, Distributions

export Bank, Firm, initialize_econ, firmbank_step!




#= TODO
1. make it so firms explore or immirate the turn after they receive the funds (maybe with another state that gets toggled: received_loan= im, ex, none and is then passed to the correct state)
2. do the capitalisation of banks, creation, default, recreation
3. do dividend payouts
=#

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
    bankrupt::Bool
    profits::Float64
end

@agent struct Firm(GridAgent{2})
    state::String # mi, im or ex
    Q::Float64 # productivity
    E::Float64 # savings
    D::Float64 # debt
    B::Float64 # bank money
    bank::Union{Bank,Nothing} # bank assigned to
    wants_loan::Bool
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

    old_pos, old_Q = firm.mem

    ############# Behaviour for Explorers ##################################
    ############# 1. Pay exploration cost #############   
    if firm.state == "ex"
        # First cost pay by bank, then by savings           
        m_old = length(agents_in_position(old_pos, model)) # nr of agents on old position
        cost = c * model.productivity[old_pos...] * (m_old)^(α - 1) # cost of exploration
        if firm.B > 0
            firm.B -= cost
            bank = firm.bank
            bank.capital -= cost
            bank.liabilities -= cost
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
        move_along_route!(firm, model, pathfinder)
        m_old = length(agents_in_position(old_pos, model))
        cost = c * old_Q * (m_old)^(α - 1)
        if firm.B > 0
            firm.B -= cost
            bank = firm.bank
            bank.capital -= cost
            bank.liabilities -= cost
        else
            firm.E -= cost
        end

        ############# 3. Check if destination reached, become miner if so #############
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
        if firm.D > 0 # if firm has debt, repay
            firm.D -= Q / (1 + r) # discount to obtain the principal
            bank.loans -= Q # outstanding loan turned into capital
            bank.capital += Q + ((Q * r) * (1 - ξ))
            bank.equity += (Q * r * (1 - ξ))
            bank.profits += (Q * r * ξ)
        else # else consume and save
            firm.E += Q * (1 - c)
            E = firm.E
            bank.capital += γ₁ * E
            bank.equity += γ₁ * E
            firm.E = (1 - γ₁) * E

        end

        ############# 2. Explorer transition, if funds, else wants_loan #############
        if rand(model.rng) ≤ ϵ # with prob ϵ
            m_pos = length(agents_in_position(firm, model))
            C = (c * Q * (m_pos)^(α - 1)) / model.π_isl #expected cost of exploration
            if (firm.E + firm.B) ≥ C
                firm.state = "ex"
                firm.mem = (firm.pos, Q)
                firm.Q = 0.0
            else
                # TODO turn bro into an explorer if she gets loan
                firm.wants_loan = true
                firm.amount = C - firm.E
            end
            ############# 2. Receive signals and become immitator #############
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
                if new_coef > old_coef # If its better than known
                    m_pos = length(agents_in_position(firm, model))
                    distance = manhattan_distance(firm.pos, new_pos, model)
                    C = (c * firm.Q * (m_pos)^(α - 1)) * distance
                    if (firm.E + firm.B) ≥ C
                        firm.state = "im" # Turn into immitator
                        firm.mem = (firm.pos, firm.Q)
                        firm.Q = 0.0 # No output
                        firm.target = new_pos #Target island
                        plan_route!(firm, new_pos, pathfinder)
                    else
                        # TODO turn bro into an imitator if she gets loan
                        firm.wants_loan = true
                        firm.amount = C - firm.E
                    end
                end
            end
        end
    end
end

function firmbank_step!(bank::Bank, model)
    Χ = model.Χ
    α = model.α

    div = bank.profits / length(bank.firms)

    ############# 1. Process defaults and pay dividends #############
    for firm in bank.firms
        if firm.E + firm.B < 0
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
            firm.E += div
        end
    end
    bank.profits = 0.0
    ############# 2. Process bankrupcy #############
    #TODO bank defaults
    if bank.equity < 0
        bank.bankrupt = true
    end

    ############# 3. Select firms and provide loans #############
    firms_looking = filter(firm -> firm.wants_loan, bank.firms)
    sorted_firms_looking = sort(firms_looking, by=firm -> firm.amount)
    c_supp = (bank.equity / Χ) - bank.loans
    total_req = 0.0
    firms_supplied = []
    for firm in sorted_firms_looking
        if total_req + firm.amount > c_supp
            break
        end
        total_req += firm.amount
        push!(firms_supplied, firm)
    end
    for firm in firms_supplied
        amount = firm.amount
        firm.D += amount
        firm.B += amount
        firm.amount = 0.0
        firm.wants_loan = false

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
    dim=5001, # Always make uneven!
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
        wants_loan = false
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
    #TODO bank capitalisation
    for i in 1:n_banks
        add_agent!(Bank, model;
            state="bank",
            capital=0.0,
            loans=0.0,
            pos=(middle, middle),
            liabilities=0.0,
            equity=0.0,
            firms=firm_groups[i],
            bankrupt=false,
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

    return model
end

end
