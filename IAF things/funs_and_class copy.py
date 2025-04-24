# %%
import numpy as np
import random as seed
import networkx as nx
import copy
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# some of these are not necessary

########################
### graph generators ###
########################


def generate_competing_kh(nodes, m, p, p_inh):
    G = nx.empty_graph(m)  # start with m isolated nodes
    G = nx.DiGraph(G)
    nx.set_node_attributes(G, 0.0, 'RISK')
    repeated_nodes = list(G.nodes())  # list of the existing nodes
    source = m  # the next node is m

    def add_bidirectional_edge(graph, u, v, edge_type):
        graph.add_edge(u, v, edge_type=edge_type)
        graph.add_edge(v, u, edge_type=edge_type)

    while source < nodes:  # keep going until nodes=n
        # pick m random available nodes
        possible_targets = seed.sample(repeated_nodes, m)
        target = possible_targets.pop()
        # bidirectional excitatory edge
        add_bidirectional_edge(G, source, target, "excitatory")
        repeated_nodes.append(target)
        count = 1

        while count < m:  # adding m-1 more new links
            if seed.random() < p:  # clustering step
                neighborhood = [
                    nbr for nbr in G.neighbors(target)
                    if not G.has_edge(source, nbr) and nbr != source
                ]
                if neighborhood:
                    nbr = seed.choice(neighborhood)
                    # bidirectional inhibitory edge
                    add_bidirectional_edge(G, source, nbr, "excitatory")
                    repeated_nodes.append(nbr)
                    count += 1
                    continue
            target = possible_targets.pop()  # preferential attachment step
            # bidirectional excitatory edge
            add_bidirectional_edge(G, source, target, "excitatory")
            repeated_nodes.append(target)
            count += 1

        repeated_nodes.extend([source] * m)
        source += 1

    edges = list(G.edges())
    num_inhibitory = int(p_inh * len(edges))
    inhibitory_edges = seed.sample(edges, num_inhibitory)
    for u, v in inhibitory_edges:
        G.edges[u, v]['edge_type'] = "inhibitory"

    nx.set_node_attributes(
        G, {i: {'RISK': 0.0, 'in_inhib': 0, 'in_excit': 0, 'FIRE': 0} for i in range(nodes)})

    for node in G.nodes:
        for neighbor in G.neighbors(node):
            edge_type = G.edges[node, neighbor].get('edge_type', 'excitatory')
            if edge_type == "inhibitory":
                G.nodes[node]['in_inhib'] += 1
            elif edge_type == "excitatory":
                G.nodes[node]['in_excit'] += 1

    for node in G.nodes:
        for neighbor in G.neighbors(node):
            edge_type = G.edges[node, neighbor].get('edge_type')
            if edge_type == "inhibitory":
                G.edges[node, neighbor]['weight'] = - \
                    1 / G.nodes[node].get('in_inhib')
            else:
                G.edges[node, neighbor]['weight'] = 1 / \
                    G.nodes[node].get('in_excit')
    return G


def n_gopoly(h, n):
    G = nx.complete_graph(h, create_using=nx.DiGraph)  # n competing firms

    nx.set_edge_attributes(G, "inhibitory", "edge_type")

    for i in range(h, n):  # n firms dependent on all competitors
        G.add_node(i)
        for j in range(h):
            G.add_edge(i, j, edge_type="excitatory")
            G.add_edge(j, i, edge_type="excitatory")

    nx.set_node_attributes(
        G, {i: {'RISK': 0.0, 'in_inhib': 0, 'in_excit': 0, 'FIRE': 0} for i in range(n)})

    for node in G.nodes:
        for neighbor in G.neighbors(node):
            edge_type = G.edges[node, neighbor].get('edge_type')
            if edge_type == "inhibitory":
                G.nodes[node]['in_inhib'] += 1
            elif edge_type == "excitatory":
                G.nodes[node]['in_excit'] += 1

    for node in G.nodes:
        for neighbor in G.neighbors(node):
            edge_type = G.edges[node, neighbor].get('edge_type')
            if edge_type == "inhibitory":
                G.edges[node, neighbor]['weight'] = - \
                    1 / G.nodes[node].get('in_inhib')
            else:
                G.edges[node, neighbor]['weight'] = 1 / \
                    G.nodes[node].get('in_excit')

    return G


def unconnected(n):
    G = nx.empty_graph(n)
    nx.set_node_attributes(G, 0.0, 'RISK')
    return G


####################################################################################################################################

    ########################
    ### simulation steps ###
    ########################

def risk_update(graph, theta):
    # .. Assign at each update a new RISK to each firm,
    # and set their FIRE score, which tracks after how many propagations it was activated, to 0
    # the FIRE attribute is just to keep track of which order of contagion the spike was in

    for j in graph.nodes:
        graph.nodes[j]['FIRE'] = 0
        graph.nodes[j]['RISK'] = np.random.beta(2.5, 5)

    # .. Spikes can travel over edges only once, and are then deactivated, stored in this set
    inactive_edges = set()

    converged = False  # .. The next loop keeps checking is neighbours have defaulted and computing the new RISK, until no node changes for a full iteration
    loop = 0  # .. to keep track in which order of propagation we are in
    while not converged:
        converged = True  # .. is set to false once a full iteration without updates happens
        loop += 1

        # .. The next loop updates risk values based on neighbours RISKS
        for j in graph.nodes:
            previous_risk = graph.nodes[j].get('RISK')
            updated_risk = previous_risk

            # If the node hasnt spiked before, note the iteration that triggered it
            if updated_risk >= theta and graph.nodes[j]['FIRE'] == 0:
                graph.nodes[j]['FIRE'] = loop
                # print(f"node {j} FIRE={loop}")

            for k in graph.neighbors(j):
                if (j, k) in inactive_edges:  # if the edge has already facilitated a spike, skip
                    continue

                # .. The next loop checks if the neighbour is spiking, and computes new RISK based on edge weight (computed in net gen functn)
                if graph.nodes[k].get('RISK') >= theta:
                    w_jk = graph.edges[j, k].get('weight')
                    updated_risk = updated_risk + w_jk * graph.nodes[k]['RISK']
                    # print(f" upd {j} dt {k} with weight {graph.edges[j, k]['weight']} from {previous_risk} to {updated_risk} ")
                    inactive_edges.add((j, k))
                    # add the edges to the inactive set after use
                    inactive_edges.add((k, j))

            if updated_risk != previous_risk:
                graph.nodes[j]['RISK'] = updated_risk
                converged = False  # is the RISK value changed, update it and set converged is false so another iteration happens

        if converged:
            break  # stop the loop if there were no updates

    return graph


def initialize(banks=1, n=100, m=3, p=0.85, p_inh=0.5, f_hist=100, theta=0.75, CB_rate=0.01):  # tested

    G = generate_competing_kh(n, m, p, p_inh)

    # Rows: firms, Columns: time steps
    equity_trajectories = np.zeros((n, f_hist))
    for t in range(f_hist):
        G = risk_update(G, theta)
        for j in G.nodes:
            equity_trajectories[j, t] = theta - G.nodes[j].get('RISK')

    for bank in range(banks):
        exec(f'BANK_{bank} = Bank(id={
             bank}, i=300000, s=300000, cash=400000, b=600000, c=400000, CB_rate=CB_rate)')

    all_banks = [eval(f'BANK_{bank}') for bank in range(banks)]

    return equity_trajectories, all_banks, G


# before this there should be some dict with the offer action of each bank
# a dict like {loan: {bank: offer}} where offer is the interest rate

def Market_Process(bank_offers, looking_to_loan):
    for firm in bank_offers:
        if looking_to_loan.get(firm):
            loan = looking_to_loan.get(firm)
            offers = []
            print(offers)
            for bank in bank_offers[firm]:
                offers.append(bank_offers[firm][bank])
                print(bank_offers[firm][bank])
            print(offers)

            print(offers)
            winner_bank = min(offers)
            print(winner_bank)
            lowest_rate = offers[winner_bank]

            if lowest_rate <= loan.i:
                loan.i = lowest_rate
                winner_bank.new_loans.append(loan)


def update_equities(G, theta, equity_trajectories, n):  # tested

    G_t = risk_update(G, theta)
    equity_trajectories = np.hstack((equity_trajectories, np.zeros((n, 1))))

    for j in G.nodes:
        equity_trajectories[j, -1] = theta - G.nodes[j].get('RISK')

    return equity_trajectories


def get_current_equity(equity_trajectories, j):  # tested
    return equity_trajectories[j, -1]


def get_equity_history(equity_trajectories, j):  # tested
    return equity_trajectories[j, : -1]


def update_banks(all_banks, CB_rate=0.01, equity_trajectories=None):  # tested
    for bank in all_banks:
        bank.update(CB_rate, equity_trajectories)
    return all_banks


def new_loans(equity_trajectories, all_banks, n, f_hist):

    # select requesting firms
    # for multiple firms:  m = np.clip(np.random.normal(0.055, 0.02), 0.01, 0.10),  looking_to_loan = {key: None for key in np.random.choice(n, int(n*m), False)}

    bank_ids = []
    for bank in all_banks:
        bank_ids.append(bank.id)

    bank_state = {bank: {firm: [] for firm in range(n)} for bank in bank_ids}

    for bank in all_banks:
        for firm in range(n):

            bank_array = np.array([bank.i,
                                   bank.s,
                                   bank.e,
                                   bank.cash,
                                   bank.a,
                                   bank.b,
                                   bank.d,
                                   bank.c,
                                   bank.l,
                                   bank.eq_cap_ratio,
                                   int(bank.failed)])

            loan_array = (np.array([0.0, 0.0]),
                          np.zeros(f_hist-1))

            bank_state[bank.id][firm] = (loan_array, bank_array)

        # now fill it with the t specific info

    min_loans = max(int(n*0.01), 1)
    max_loans = max(int(n*0.05), 1)

    looking_to_loan = random.sample(range(n), random.randint(
        min_loans, max_loans))  # between 1% and 5% of firms come borrow

    created_loans = {}

    for firm in looking_to_loan:

        # NUMBERS TO TWEAK
        # loan amount
        x = int(np.clip(np.random.normal(50000, 20000), 1000, 100000))
        i = 0.2  # max interest rate firm accepts
        m_t = np.random.randint(6, 60)  # random maturity
        b = np.random.randint(2, 5)  # to how many banks does the firm go to

        # each firm creates a requested loan
        loan_created = Loan(
            j=firm, equity_trajectories=equity_trajectories, x=x, i=i, m_t=m_t)
        created_loans[firm] = loan_created

        # which banks does the firm visit?
        banks_visited = random.sample(range(len(bank_ids)), random.randint(
            0, len(bank_ids)))

        # retrieve bank objects from ID (because the observation space doesnt like bank object)
        for bank in banks_visited:
            obj_visited = None
            for obj in all_banks:
                if obj.id == bank:
                    obj_visited = obj  # This is the bank obj for current bank

            #  bank agent i should get as observation: all bank_state[i][firm] that are non empty

            bank_array = np.array([obj_visited.i,
                                   obj_visited.s,
                                   obj_visited.e,
                                   obj_visited.cash,
                                   obj_visited.a,
                                   obj_visited.b,
                                   obj_visited.d,
                                   obj_visited.c,
                                   obj_visited.l,
                                   obj_visited.eq_cap_ratio,
                                   int(obj_visited.failed)])

            loan_array = (np.array([loan_created.x, loan_created.m_t]),
                          loan_created.hist_t[-f_hist:])

            bank_state[bank][firm] = (loan_array, bank_array)

    # check what haoppens if the same bank gets two offers in a period
    return bank_state, created_loans


def gen_step(nodes, m, p, p_inh, theta):
    graph = generate_competing_kh(nodes, m, p, p_inh)
    result = risk_update(graph, theta)
    return result


def episode(nodes, m, p, p_inh, theta, t):
    for t in range(t):
        graph_snap = []
        graph = gen_step(nodes, m, p, p_inh, theta)
        graph_snap.append(copy.deepcopy(G))
        return graph_snap


############################################################################################################################################################################################################################
    #############
    ### plots ###
    #############

def plot_spikes(graph_snap, theta):

    # Create a figure and an axes to plot the spike occurrences
    fig, ax = plt.subplots(figsize=(15, 10))

    # Set the background color of the figure and axes to white
    # fig.patch.set_facecolor('white')
    # ax.set_facecolor('white')

    # Define a colormap and normalize for FIRE values between 1 and 10
    cmap = cm.get_cmap('plasma', 6)  # You can choose any colormap you like
    norm = mcolors.Normalize(vmin=1, vmax=3)

    # Iterate over each snapshot at each time step
    for t_idx, G in enumerate(graph_snap):
        for node in G.nodes:
            x = t_idx  # Time index on x-axis
            y = node   # Node index on y-axis

            risk = G.nodes[node]['RISK']
            fire = G.nodes[node].get('FIRE', None)  # Safely get the FIRE value

            if risk >= theta:
                if fire is not None:  # Only plot if FIRE is defined
                    # Get the color for the current FIRE value
                    color = cmap(norm(fire))

                    # Plot the spike with the corresponding color
                    ax.plot(x, y, '^', markersize=6, color=color)

    # Set axis labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Firm Index')
    ax.set_title('Default Occurrences Over Time')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add a colorbar to show the mapping of colors to FIRE values
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for older versions of Matplotlib
    fig.colorbar(sm, ax=ax, ticks=range(1, 11), label='Order of Spillover')

    plt.show()

#####################################################################################################################################
    ###############
    ### Classes ###
    ###############


class Loan:  # tested
    # equity_trajectories should be a (very) long array of all the equity time series of all firms. Can be computed beforehand since agents dont affect it
    def __init__(self, j, equity_trajectories, x, i, m_t):

        # initial loan i should be the max rate the firm accepts
        # info on firm and time
        self.j = j  # Firm the loan was granted to
        self.E = get_current_equity(
            equity_trajectories, self.j)  # Current equity
        # time series of equity up to t
        self.hist_t = get_equity_history(equity_trajectories, self.j)

        # info on loan specifics
        self.x = x  # loan amount
        self.i = i  # interest on loan
        self.m_t = m_t  # remaining maturity on loan
        self.r_t = i*x  # revenue to bank payed over loan

        self.terminal = False  # Check if loan is at maturity
        self.defaulted = False

    # should be done after each time step as part of state transition independent of action
    def update(self, equity_trajectories):

        # update risk info
        self.E = get_current_equity(
            equity_trajectories, self.j)  # Current equity
        # equity time series most recent
        self.hist_t = get_equity_history(equity_trajectories, self.j)

        # check default
        if self.E <= 0:
            self.defaulted = True
            print(f"Loan {self.j} defaulted!")

        # for non-mature loans
        if self.m_t > 0:
            self.m_t = self.m_t-1  # reduce maturity by 1 period

            # maturity
        if self.m_t == 0:  # is loan is at maturity
            self.terminal = True  # Bank class should remove loan if terminal = True
            print(f"Loan {self.j} matured!")


class Bank:  # tested
    def __init__(self, id=0, i=0, s=0, cash=0, b=0, c=0, CB_rate=0):

        self.id = id
        self.portfolio = []
        self.new_loans = []

        # Assets
        self.i = i
        self.s = s
        self.e = 0
        self.cash = cash
        self.a = self.i + self.s + self.e + self.cash
        # Liabilities
        self.b = b
        self.d = 0
        self.c = c
        self.l = self.b + self.d + self.c
        self.CB_rate = CB_rate

        self.eq_cap_ratio = (self.b+self.d)/self.c
        self.failed = False

        self.bal = self.a - self.l

    def update(self, CB_rate, equity_trajectories):  # CB_rate < 1

        for loan in self.portfolio:
            loan.update(equity_trajectories)

        # add new loans to balance sheet and portfolio
        for new_loan in self.new_loans:
            self.e = self.e + new_loan.x
            self.d = self.d + new_loan.x
            self.portfolio.append(new_loan)
        self.new_loans = []  # empty the list

        # process defaulted, matured and interest
        remove = []

        for loan in self.portfolio:
            if loan.defaulted:
                self.e = self.e - loan.x
                remove.append(loan)
                print(f"Processing default of {loan.j}")
            elif loan.terminal:
                self.d = self.d - loan.x
                self.e = self.e - loan.x
                remove.append(loan)
            else:
                self.cash = self.cash + loan.r_t
                self.c = self.c + loan.r_t

        for loan_delete in remove:
            self.portfolio.remove(loan_delete)
        remove = []  # empty the list

        # pay CB deposit rate
        cash_lost = self.cash * CB_rate
        self.cash = self.cash - cash_lost
        self.c = self.c - cash_lost

        # recalculate total assets and liabilities
        self.a = self.i+self.s+self.e+self.cash
        self.l = self.b + self.d + self.c

        # Absorb losses with equity
        if self.l >= self.a:
            self.c = self.c - self.l + self.a

        # check default
        if self.c <= 0:
            self.failed = True
