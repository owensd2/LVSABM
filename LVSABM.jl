##########################
# Land Value Sharing ABM #
##########################

# This model simulates the effect of Land Value Sharing on the number of transactions which 
# occur in a property/development market and the prices at which they occur

# If you have any questions please don't hesitate to ask
# dowens@indecon.ie 
# Daniel Owens, Indecon International Economic consultants

# Copyright (c) 2022, Daniel Owens, Indecon International Economic Consultants
# All rights reserved.

using Pkg

Pkg.add(["Agents", "Random", "StatsBase", "Distributions", "Dates", "GLMakie", "Makie", "Plots", "Colors"])

using Agents, Random, StatsBase, Distributions, Dates

# Buyer agent and associated values
@agent Buyer GridAgent{2} begin
    buyer::Bool         # is this agent a buyer
    traded::Bool        # has this agent already been involved in a transaction
    budget::Float64     # the 'budget'
    utility::Float64    # utility of the property its bid is for
    my_price::Float64   # value of its bid
    alpha::Float64      # preference for proximity to ammenities over proximity to urban centre(s)
    b::Float64          # parameter for how much increased utility affects WTP (and so my_price)
end

# Seller agent and associated values
@agent Seller GridAgent{2} begin
    buyer::Bool                 # is this agent a buyer
    traded::Bool                # has this agent already been involved in a transaction
    my_price::Float64           # value of ask price
    offer_list::Vector{Buyer}   # list of buyers submitting bids to this seller
end

# distance from p to q
d(p, q) = sqrt(sum((p-q).^2))

# Beta distribution with mean μ and standard deviation σ
function BetaMSD(μ, σ)
    α = (μ^2 - μ^3)/(σ^2) - μ
    β = (μ-1)*(μ^2-μ)/var + μ - 1
    return Beta(α, β)
end

# model initialzation function
# keyword arguments types and default values

# todo heterogeneity of developers - sample from distribution? - sample from different types?
# todo allow developers to buy more than 1 plot of land?
# todo monte carlo methods?
function initialize(; 
    dims                    = (29,29),                          # dimensions of grid
    centre                  = [[15,15]],                        # location(s) of urban centre(s)
    greens                  = [],                               # locations of ammenities
    num_buyers::Int         = 841,                              # number of buyers
    num_sellers::Int        = prod(dims) - length(greens),      # number of sellers
    ex_price::Float64       = 200.0,                            # value of grid square currently 
    ex_markup::Float64      = 0.25,                             # seller markup rate
    budget::Float64         = 800.0,                            # buyer 'budget'
    alpha::Float64          = 0.15,                             # preference for proximity to ammenities over proximity to urban centre(s) (∈ [0,1])
    seed::Int               = millisecond(now())*second(now()), # seed for random number generation (allows for reproducibilty)
    num_viewed::Int         = 10,                               # number of sites view by each seller per time step
    TCU::Float64            = 1.0,                              # costs from distance to urban centre
    b::Float64              = 70.0,                             # parameter for how much increased utility affects WTP
    num_active_buyers::Int  = num_buyers,                       # number of active buyers per time step
    tax_rate::Float64       = 0.0,                              # rate of tax
    negotiation::String     = "am",                             # how transaction price is calculated from bid and ask (am = arithmetic mean, gm = geometric mean)
    weight::Float64         = 0.5,                              # weight for bid if negotiation is weighted (∈ [0,1])
    proximity::String       = "default"                         # how proximity is calculated ('default' is normalised from 0 to 1), 'inverse' is 1/distance
    )

    if weight > 1 || weight < 0
        error("weight is outside of range [0, 1]")
    end

    if alpha > 1 || alpha < 0
        error("alpha is outside of range [0, 1]")
    end

    space = GridSpace(dims, periodic = false, metric = :euclidean)

    rng = Random.MersenneTwister(seed)

    if lowercase(negotiation) == "am"
        negotiate(a, b) = weight*a + (1-weight)*b
    elseif lowercase(negotiation) == "gm"
        negotite(a,b) = a^weight * b^(1-weight)
    else
        error("the only valid inputs are: 'am' and 'gm'")
    end

    # The `area` field is coded such that Empty = 0, Greens = 1, centre = 2
    # The `state` field is coded such that Empty = 0, For sale = 1, Sold = 2
    properties = Dict(
        :num_buyers         => num_buyers,
        :num_sellers        => num_sellers <= prod(dims) - length(greens) ? num_sellers : error("more sellers than plots"),
        :dist_centre        => zeros(Float64, dims),
        :dist_green         => zeros(Float64, dims),
        :p_centre           => zeros(Float64, dims),
        :p_green            => zeros(Float64, dims),
        :area               => zeros(Int, dims),
        :state              => zeros(Int, dims),
        :price              => zeros(Float64, dims),
        :WTA                => ex_price * (1 + ex_markup),
        :ex_price           => ex_price,
        :epsilon            => (num_buyers - num_sellers)/(num_buyers + num_sellers),
        :buyers             => [],
        :sellers            => [],
        :TCU                => TCU,
        :num_viewed         => num_viewed,
        :num_active_buyers  => num_active_buyers,
        :active_buyers      => [],
        :tax_rate           => tax_rate,
        :tax_revenue        => zeros(Float64, dims),
        :negotiate          => negotiate)

    model = ABM(Union{Buyer, Seller}, space; properties, rng)

    for I in greens
        model.area[I[1], I[2]] = 1
    end

    for I in centre
        model.area[I[1], I[2]] = 2
    end

    for i in 1:dims[1] 
        for j in 1:dims[2]
            if length(centre) > 0
                model.dist_centre[i,j] = minimum([d([i,j], c) for c in centre])
            else
                model.dist_centre[i,j] = 1
            end
        end
    end
    model.p_centre = -model.dist_centre .+ 1 .+ maximum(model.dist_centre)
    model.p_centre ./= maximum(model.p_centre)

    for i in 1:dims[1] 
        for j in 1:dims[2]
            if length(greens) > 0
                model.dist_green[i,j] = minimum([d([i,j], c) for c in greens])
            else
                model.dist_green[i,j] = 1
            end
        end
    end

    if lowercase(proximity) == "default"
        model.p_green = -model.dist_green .+ 1 .+ maximum(model.dist_green)
        model.p_green ./= maximum(model.p_green)

        model.p_centre = -model.dist_centre .+ 1 .+ maximum(model.dist_centre)
        model.p_centre ./= maximum(model.p_centre)
    elseif lowercase(proximity) == "inverse"
        if length(greens) ==0
            model.p_green .= 1
        else
            s_green = (0.5*maximum(model.dist_green) + 2) / (log(maximum(model.dist_green)) + 1)
            c_green = (1 - s_green)/(maximum(model.dist_green))
            model.p_green = s_green ./ (model.dist_green .+ eps()) .+ c_green
        end
        if length(centre) == 0
            model.p_centre .= 1
        else
            s_centre = (0.5*maximum(model.dist_centre) + 2) / (log(maximum(model.dist_centre)) + 1)
            c_centre = (1 - s_centre)/(maximum(model.dist_centre))
            model.p_green = s_centre ./ (model.dist_centre .+ eps()) .+ c_centre
        end
    else
        error("the only valid inputs are: 'default' and 'inverse'")
    end

    spots = [I for I in CartesianIndices(model.area) if model.area[I] in [0,2]]

    for i in 1:(num_buyers+num_sellers)
        if i <= num_sellers
            j = rand(1:length(spots))
            spot = spots[j]
            agent = Seller(i, spot, false, false, model.WTA*(1+model.epsilon), [])
            model.state[spot] = 1
            model.sellers = vcat(model.sellers, [agent])
            deleteat!(spots,j)
        else
            spot = (1,1)
            agent = Buyer(i, spot, true, false, budget, 0, 0, alpha, b)
            model.buyers = vcat(model.buyers, [agent])
        end
        add_agent!(agent, (spot[1], spot[2]), model)
    end

    return model
end

# sampling without replacement, returns whole set if sample size is greater than set size
my_sample(rng, list, n) = length(list) >= n ? sample(rng, list, n, replace = false) : list

# utility function
util(model, agent, position) = 100 * (model.p_centre[position]^(1-agent.alpha) * model.p_green[position]^agent.alpha)

# actions of each agent at each time step
function agent_step!(agent, model)
    if agent.traded
        return
    end
    if !agent.buyer
        if length(agent.offer_list) > 0
            offer_price, i_best = findmax([buyer.my_price for buyer in agent.offer_list])
            if offer_price >= agent.my_price
                price = (offer_price + agent.my_price) / 2
                spread = price - model.ex_price
                taxes = spread * model.tax_rate
                if offer_price - taxes >= agent.my_price
                    agent.traded = true
                    best = agent.offer_list[i_best]
                    best.traded = true
                    model.price[CartesianIndex(agent.pos)] = model.negotiate(offer_price, agent.my_price)
                    model.state[CartesianIndex(agent.pos)] = 2
                    model.num_buyers -= 1
                    model.num_sellers -= 1
                    model.sellers = [seller for seller in model.sellers if !seller.traded]
                    model.buyers = [b for b in model.buyers if !b.traded]
                end
            end
            agent.offer_list = []
        end
        agent.my_price = model.WTA * (1 + model.epsilon)
    elseif agent.buyer && agent.id in model.active_buyers
        sellers = [seller for seller in my_sample(model.rng, model.sellers, model.num_viewed)]
        utils = [util(model, agent, CartesianIndex(seller.pos)) for seller in sellers]
        best_util, i_best = findmax(utils)
        best = sellers[i_best]
        Y = agent.budget - (model.dist_centre[CartesianIndex(best.pos)] * model.TCU)
        agent.my_price = (Y * best_util^2 * (1 + model.epsilon)) / (agent.b^2 + best_util^2)
        move_agent!(agent, best.pos, model)
        best.offer_list = vcat(best.offer_list, [agent])
    end
end

# updating values in the model at each time step
function model_step!(model)
    model.epsilon = (model.num_buyers - model.num_sellers)/(model.num_buyers + model.num_sellers)
    model.active_buyers = [buyer.id for buyer in my_sample(model.rng, model.buyers, model.num_active_buyers)]
end


