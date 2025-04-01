module GrowthBasic

using Agents, Agents.Pathfinding, Random, Distributions



##############
### Agents ###
##############

@agent struct Firm(GridAgent{2})
    state::String
    Q::Float64
    mem::Float64
    target::Tuple{Int,Int}
end

######################
### Step Functions ###
######################
function firm_step!(firm::Firm, model)

    # Take params from model
    α = model.α
    ϵ = model.ϵ
    λ = model.λ
    ϕ = model.ϕ
    ρ = model.ρ
    middle = model.middle # Where the agents start
    pathfinder = model.pathfinder  #Pathfinder algo finds shortest paths

    ### Behaviour for Explorers
    if firm.state == "ex"
        randomwalk!(firm, model, 1) #Take one random-walk step
        position = firm.pos
        if model.island_dummy[position...] == true #If they find an island
            firm.state = "mi" # Become Miner
            if model.disc_dummy[position...] == false ## If it was undescovered
                model.disc_dummy[position...] == true
                #Get vars for the new island productivity
                W = rand(model.rng, Poisson(λ)) # Get lucky sometimes
                ω = rand(model.rng, Normal(0, 1)) # Noise
                x = abs(position[1] - middle) # Distances from "origin"
                y = abs(position[2] - middle) # Take out middle to adjust for starting in centre of matrix
                # New island productivity coefficient
                model.productivity[position...] = (1 + W) * (x + y + ϕ * firm.mem + ω)
                # Firm productivity updated   
            end
            m_pos = length(agents_in_position(firm, model))
            firm.Q = model.productivity[position...] * (m_pos)^(α - 1)
        end

        ### Behaviour for Immitators
    elseif firm.state == "im"
        move_along_route!(firm, model, pathfinder)
        position = firm.pos
        if firm.pos == firm.target
            firm.state = "mi"
            m_pos = length(agents_in_position(firm, model))
            firm.Q = model.productivity[position...] * (m_pos)^(α - 1)
        end

        ### Behaviour for Miners
    elseif firm.state == "mi"
        position = firm.pos
        # Miners receive signals, or randomly turn into explorers
        mi_agents = filter(agent -> agent.state == "mi", collect(allagents(model))) # List of all miners
        other_islands = filter(agent -> agent.pos != position, mi_agents) # List of miners on other islands

        signals_received = Dict((middle, middle) => model.productivity[position...]) #

        if rand(model.rng) ≤ ϵ
            firm.state = "ex"
            firm.mem = firm.Q
            firm.Q = 0.0
        else
            for miner in other_islands
                # All miners on island where signal from
                mⱼ = length(filter(agent -> agent.state == "mi", collect(agents_in_position(miner.pos, model))))
                m = length(mi_agents) #Total miners
                distance = manhattan_distance(firm, miner, model)
                wⱼ = (mⱼ / m) * exp(-ρ * distance)
                if mⱼ > m
                    print(mⱼ - m)
                    error("Process interrupted: this cant be!!!!")
                end
                received = rand(model.rng, Binomial(1, wⱼ)) #Chance that signal is received
                if received == 1 # if the signal was received
                    signals_received[miner.pos] = model.productivity[miner.pos...]
                end
            end
            new_coef, new_pos = findmax(signals_received) # Select the best signal
            old_coef = model.productivity[position...]
            if new_coef > old_coef # If its better than known
                firm.state = "im" # Turn into immitator
                firm.Q = 0.0 # No output
                firm.target = new_pos #Target island
                plan_route!(firm, new_pos, pathfinder)
            end
        end
    end
end

######################
### Initialization ###
######################

function initialize_econ(;
    n_firms=100,
    dim=3001, # Always make uneven!
    π_isl=0.1,
    α=1.5,
    ϵ=0.1,
    λ=1,
    ϕ=0.5,
    ρ=0.1,
    seed=42)

    middle = dim ÷ 2 + 1 # Selects the middle of the uneven dim, so that we spawn all the agents in the center of the space
    rng = MersenneTwister(seed)

    # Create the technology space
    tech_space = GridSpace((dim, dim); periodic=false, metric=:manhattan) #Manhattan as per the paper, periodic turns it into a donut, we just make it larger than the agents can reach within T
    pathfinder = AStar(tech_space; diagonal_movement=false) # Specify which algo finds the shortest path for the immitator agents

    properties = (
        island_dummy=falses(dim, dim), # We will set some to True based on prob of island
        disc_dummy=falses(dim, dim),
        productivity=ones(dim, dim), # New prods are set once the islands are discovered
        α=α,
        ϵ=ϵ,
        λ=λ,
        ϕ=ϕ,
        ρ=ρ,
        middle=middle,
        pathfinder=pathfinder,
        rng=rng
    )

    #Initialize model
    model = StandardABM(Firm, tech_space; properties=properties, (agent_step!)=firm_step!, rng=rng,
        scheduler=Schedulers.fastest,
        warn=true)

    #Adding agents
    for _ in 1:n_firms
        state = "mi"
        Q = 0.0
        mem = 0.0
        pos = (middle, middle)
        add_agent!(Firm, model; state=state, Q=Q, mem=mem, pos=pos, target=pos)
    end

    #Generating islands
    for p in positions(model)
        model.island_dummy[p...] = rand(model.rng) ≤ π_isl
    end

    for agent in allagents(model)
        agent.Q = (n_firms)^(α - 1)
    end

    model.disc_dummy[middle, middle] = true

    return model
end

end



