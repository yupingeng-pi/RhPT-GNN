# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import pandas as pd
import pickle
from matpower import start_instance
import random
from time import time
import contextlib
import os
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_mean = 0.0
noise_std = 0.01
m = start_instance()
# To make Matpower run silently (ofr function inferrence)
# current_directory = os.path.dirname(os.path.abspath(__file__))
# m.eval(f"addpath('{current_directory}');")

"""
This is the main preprocessing function. It uses Octave to run matpower. It may take some tweaking to work.

The way it works is the following: 
    1. Extract the characteristics of the grid. These are elements independent of the demand such as G and B
    2. Create M input demands by applying a uniform distribution on active and reactive power demand
    3. For each input, solve the AC-OPF using matpower
    4. Regroup all of the input data into training, validation and test sets
    5. Save everything

List of elements that are classified in the 'characteristics' dictionnary:
characteristics = {'gen_index': gen_index,'gen_groups': gen_groups, 'gen_to_bus': gen_to_bus, 'edge_to_bus': edge_to_bus, 'static_costs': static_costs}

    - node_nb = number of real buses
    - total_node_nb = number of real buses + artificial buses
    - nodes = Dataframe that contains costs and bus type
    - node_limits = Df that contains the limits on all the nodes, i.e: Vmin, Vmax, Pmin, Pmax, Qmin, Qmax
    - edge_limits = Df with the edge transmission limits, shunts and tap ratios
    - actual_edges = all the edge indices without the self loops (the version with self loops is given to the GNN under the name edge_index)
    - G = list of edge conductances
    - B = list of edge impendances
    - norm_coeffs = normalization coefficients
    - Ref_node = index of the reference node. If it doesn't exist it defaults to -1
    - gen_index = short list of generator indices
    - gen_groups = dictionnary that keeps track of which artificial node is linked with which real bus (obsolete)
    - gen_to_bus = sparse matrix that says which generator is attached to which bus
    - edge_to_bus = consists of 2 matrices, edge_to_bus and edge_from_bus
    - static_costs = costs of order 0 (don't depend on value of generated power)
"""

# case = 'case30'
# case = "case9"
# case = "case24_ieee_rts"
# case = 'case9Q'
case = 'case118'
# case = 'case145'

points = 500
batch = 20
spread = 0
split = [0.8, 0.1, 0.1]

# Topology perturbation parameters according to paper Table 2
perturbation_params = {
    'case9': {'p': 0.05, 'K': 20},
    'case24_ieee_rts': {'p': 0.03, 'K': 45},
    'case30': {'p': 0.03, 'K': 60},
    'case118': {'p': 0.012, 'K': 200}
}
# Get perturbation parameters for current case
p_perturb = perturbation_params.get(case, {'p': 0.05, 'K': 20})['p']
K_max = perturbation_params.get(case, {'p': 0.05, 'K': 20})['K']
# split = [0,0,1] #This split serves to generate test samples
clean = True  # This determines whether or not we clean the input dataset, i.e: remove the input demand samples that matpower considers not being "successful"
mpc = m.loadcase(case)

Ybus = m.makeYbus(mpc['baseMVA'], mpc['bus'], mpc['branch'])
edge_index = [list(Ybus.indices), sorted(Ybus.indices)]

# Any potential modifications made to the case -----------------------------------------------------------------

# Case Name modifiers:

# case += '_test'
# case += '_N-1'

# Code to remove a branch:

# mpc['branch'] = np.delete(mpc['branch'], (7), axis=0)
# mpc['branch'][7][5] = 0
# mpc['branch'][7][6] = 0
# mpc['branch'][7][7] = 0
# mpc['branch'][7][10] = 0


# Step 1: Edges ------------------------------------------------------------------------------------------------

nb_nodes = len(mpc['bus'])
nb_edges = len(mpc['branch'])
nb_gens = len(mpc['gen'])
Base_MVA = mpc['baseMVA']

avg_inference_time = []

edge_limits = []

actual_edges = []
G = []
B = []
edge_to_bus = np.zeros((2, 2 * nb_edges, nb_nodes))  # Can change this to 2,nb_edges
edge_counter = 0
for line in mpc['branch']:
    if line[8] == 0:
        tap = 1
    else:
        tap = line[8]
    Y = 1 / ((line[2] + 1j * line[3]) * tap)

    if line[5] == 0:  # Case where there is no transmission line limit
        line[5] = 10000
    G.append(np.real(Y))
    B.append(np.imag(Y))
    actual_edges.append([int(line[0]) - 1, int(line[1]) - 1])
    edge_limits.append([line[5], line[4], 1 / tap])  # This isn't correct in the original code. It isn't symmetric

    edge_to_bus[0][edge_counter][int(line[0]) - 1] = 1
    edge_to_bus[1][edge_counter][int(line[1]) - 1] = 1

    G.append(np.real(Y))
    B.append(np.imag(Y))
    actual_edges.append([int(line[1]) - 1, int(line[0]) - 1])
    edge_limits.append([line[5], line[4], tap])  # S_limit, shunt, tap

    edge_to_bus[1][edge_counter + 1][int(line[0]) - 1] = 1
    edge_to_bus[0][edge_counter + 1][int(line[1]) - 1] = 1

    edge_counter += 2

# So far this only counts the original branches with self loops

gen_to_bus = np.zeros((nb_nodes, nb_nodes))

edge_to_bus = torch.from_numpy(edge_to_bus).type(torch.float32),

# Step 2: Nodes -------------------------------------------------------------------------------
nodes = {}
node_limits = {}

for i in range(nb_nodes):
    nodes[i] = {'type': 0, 'cp_1': 0, 'cp_2': 0, 'cp_3': 0, 'cq_1': 0, 'cq_2': 0,
                'cq_3': 0}  # Node level characteristics --> mostly all the costs
    node_limits[i] = {'P_max': 0, 'P_min': 0, 'Q_max': 0, 'Q_min': 0, 'V_max': mpc['bus'][i][11],
                      'V_min': mpc['bus'][i][12]}

original_structure = nodes.copy()

# Step 3: distribute all of the info into the correct dictionnaries -----------------------------

seen_gens = []
gen_index = []
gen_groups = {}
count = 0
original_gen_count = 0
Q_case = False
if len(mpc['gencost']) == 2 * nb_gens:
    Q_case = True
for j in range(len(mpc['gen'])):
    node = int(mpc['gen'][j][0]) - 1
    # if False: #By commenting the line below and uncommenting this one you can get Tom's version of node splitting
    if node not in seen_gens:  # This is the case where theres a single generator for that node
        # print(mpc['gen'][j])
        nodes[node]['cp_1'] = mpc['gencost'][j][4]
        nodes[node]['cp_2'] = mpc['gencost'][j][5]
        nodes[node]['cp_3'] = mpc['gencost'][j][6]  #
        if Q_case:
            nodes[node]['cq_1'] = mpc['gencost'][j + nb_gens][4]
            nodes[node]['cq_2'] = mpc['gencost'][j + nb_gens][5]
            nodes[node]['cq_3'] = mpc['gencost'][j + nb_gens][6]
        nodes[node]['type'] = 5  # 5 are those nodes that matter for the computation of the cost
        node_limits[node]['P_max'] = mpc['gen'][j][8]
        node_limits[node]['P_min'] = mpc['gen'][j][9]
        node_limits[node]['Q_max'] = mpc['gen'][j][3]
        node_limits[node]['Q_min'] = mpc['gen'][j][4]
        gen_groups[node] = []
        gen_index.append(node)

        gen_to_bus[node][node] = 1
        original_gen_count += 1
    else:  # Case where there's more than one gen for that given node
        nodes[node]['type'] = 4  # This corresponds to dummy nodes

        if Q_case:
            nodes[nb_nodes + count] = {'type': 5.0, 'cp_1': mpc['gencost'][j][4], 'cp_2': mpc['gencost'][j][5],
                                       'cp_3': mpc['gencost'][j][6], 'cq_1': mpc['gencost'][j + nb_gens][4],
                                       'cq_2': mpc['gencost'][j + nb_gens][5], 'cq_3': mpc['gencost'][j + nb_gens][6]}
        else:
            nodes[nb_nodes + count] = {'type': 5.0, 'cp_1': mpc['gencost'][j][4], 'cp_2': mpc['gencost'][j][5],
                                       'cp_3': mpc['gencost'][j][6], 'cq_1': 0, 'cq_2': 0, 'cq_3': 0}
        node_limits[nb_nodes + count] = {'P_max': mpc['gen'][j][8], 'P_min': mpc['gen'][j][9],
                                         'Q_max': mpc['gen'][j][3], 'Q_min': mpc['gen'][j][4], 'V_max': 0, 'V_min': 0}

        new_column = np.zeros((nb_nodes, 1))
        new_column[node] = 1

        gen_to_bus = np.concatenate((gen_to_bus, new_column), axis=1)
        # Add edge in one direction
        edge_index[1].append(nb_nodes + count)
        edge_index[0].append(node)

        # Add edge for the opposite direction
        edge_index[0].append(nb_nodes + count)
        edge_index[1].append(node)

        # Add self loops
        edge_index[0].append(nb_nodes + count)
        edge_index[1].append(nb_nodes + count)

        gen_groups[node].append(nb_nodes + count)
        gen_index.append(nb_nodes + count)
        count += 1

    seen_gens.append(node)

total_nb_nodes = len(nodes)

gen_to_bus = torch.from_numpy(gen_to_bus).type(torch.float32)

# Step 4: Generate the points -----------------------------------------

# Idea - Maybe I split the nodes only once the points have been generated and computed. --> Actually for the computation we don't even need to have anything created by then
node_inputs = []
P_distributions = []
Q_distributions = []
for i in range(nb_nodes):  # This is where we can change the distribution of inputs
    P_distrib = np.random.uniform((1 - spread) * mpc['bus'][i][2], (1 + spread) * mpc['bus'][i][2], size=points)
    P_distributions.append(P_distrib)
    Q_distrib = np.random.uniform((1 - spread) * mpc['bus'][i][3], (1 + spread) * mpc['bus'][i][3], size=points)
    Q_distributions.append(Q_distrib)
for j in range(points):
    # Step 1: Save original branch configuration
    original_branch = copy.deepcopy(mpc['branch'])

    # Step 2: Implement Algorithm 1 - Topological Random Perturbation
    perturbation_count = 0
    perturbed_lines = []
    available_lines = list(range(len(mpc['branch'])))

    # While loop continues until max perturbations K is reached or no lines available
    while perturbation_count < K_max and len(available_lines) > 0:
        r = np.random.rand()
        if r < p_perturb:
            # Randomly select a line from available lines
            idx_in_available = np.random.choice(len(available_lines))
            perturbed_branch_idx = available_lines[idx_in_available]

            # Execute topology perturbation: set S_max to 0 (simulate line outage)
            mpc['branch'][perturbed_branch_idx][5] = 0

            # Update perturbation tracking
            perturbation_count += 1
            perturbed_lines.append(perturbed_branch_idx)
            available_lines.remove(perturbed_branch_idx)

    # Step 3: Generate load samples with Gaussian noise injection
    # ============================================================
    # Node Feature Construction (related to paper Equations 26-28)
    # ============================================================
    # Paper defines different feature dimensions for different node types:
    #   - Generator nodes (Eq 26): [Pd, Qd, V_hat, theta_hat, Pg_hat, Qg_hat, eta_V, eta_P, eta_Q]
    #   - Load nodes (Eq 27): [Pd, Qd, V_hat, theta_hat, eta_V]
    #   - Line nodes (Eq 28): [V_hat, theta_hat, Z_hat, Iloss]
    #
    # Implementation simplification:
    # - All nodes use unified 6-dimensional features for efficient batching
    # - Features: [Pd, Qd, type_code, position, load_hash, topology_flag]
    # - Node type information is encoded via:
    #   1. type_code in feature dimension [2]
    #   2. node_type label used by the GNN's type embedding module
    # ============================================================
    sample = np.zeros((nb_nodes + count, 6))
    for i in range(nb_nodes):
        original_Pd = P_distributions[i][j]
        original_Qd = Q_distributions[i][j]
        # Add Gaussian noise as per Equations 19-20 in paper
        noisy_Pd = original_Pd + np.random.normal(noise_mean, noise_std)
        noisy_Qd = original_Qd + np.random.normal(noise_mean, noise_std)
        sample[i][0] = noisy_Pd
        sample[i][1] = noisy_Qd
        # Store perturbation count instead of boolean flag
        sample[i][5] = perturbation_count

    node_inputs.append([sample])

    # Step 4: Restore original topology for next iteration
    mpc['branch'] = original_branch

# """
# Step 5: Solve the inputs -----------------------------------------------------------------------

node_optimum = []
inputs_to_remove = []

for k in range(len(node_inputs)):
    sample = node_inputs[k]
    for i in range(nb_nodes):  # We change the values of the input with the new Pd and Qd
        if (sample[0][i][0] > 0 or sample[0][i][1] > 0):
            if nodes[i]['type'] < 4:  # Case where there's only load
                nodes[i]['type'] = 1.0
        mpc['bus'][i][2] = sample[0][i][0]
        mpc['bus'][i][3] = sample[0][i][1]
    mpopt = m.mpoption('verbose', 0, 'out.all', 0, 'opf.ac.solver', 'MIPS',
                       'mips.max_it', 150,
                       'mips.gradtol', 1e-8,
                       'mips.comptol', 1e-8,
                       'mips.costtol', 1e-8)
    # print(mpc)
    t0 = time()
    r = m.runopf(mpc, mpopt, nout='max_nout')
    t1 = time()
    print(r[6])
    avg_inference_time.append((t1 - t0))
    res = {'baseMVA': r[0], 'bus': r[1], 'gen': r[2], 'gencost': r[3], 'branch': r[4]}
    # At this point to distribute the accurate Pg we need to have the fullt split up node dictionnary
    sample_solution = np.zeros((total_nb_nodes, 4))  # Solution of a given sample
    gen_count = 0
    if r[
        6] == 1 or not clean:  # Checks to see if the runopf was successful, which is given by the r[6] variable (0 = failure, 1 = success)
        gen_count = 0
        for gen in gen_index:
            if nodes[gen]['type'] >= 4:
                sample_solution[gen][0] = res['gen'][gen_count][1]  # These are Pg and Pd
                sample_solution[gen][1] = res['gen'][gen_count][2]
                gen_count += 1

        for i in range(total_nb_nodes):
            if i < nb_nodes:  # This still works elegantly
                sample_solution[i][2] = res['bus'][i][7]
                sample_solution[i][3] = res['bus'][i][8]
            else:
                sample_solution[i][2] = 0
                sample_solution[i][3] = 0
        node_optimum.append([sample_solution])
    else:
        inputs_to_remove.append(k)

for index in sorted(inputs_to_remove, reverse=True):
    del node_inputs[index]

print('Size of total set before split: ')
print(len(node_optimum))
print(len(node_inputs))

ref_node = -1
for i in range(nb_nodes):
    if node_optimum[0][0][i][3] == 0:
        ref_node = i

count_batch = 0
for sample in node_inputs:
    rand = random.uniform(0.1, 1)
    for j in range(nb_nodes + count):
        Pd = sample[0][i].sum()
        if j < nb_nodes:
            sample[0][j][0] += - mpc['bus'][j][4]  # Include all of the extra injections
            sample[0][j][1] += - mpc['bus'][j][5]

        # !!! Additional input feature dimensions (implementation detail) !!!
        # These features are not explicitly mentioned in paper equations 26-28,
        # but help the model learn better representations:
        sample[0][j][2] = 0.1 * nodes[j]['type'] / 5  # Normalized node type (0-1 range)
        sample[0][j][3] = 0.1 * j / (nb_nodes + count)  # Normalized node position
        sample[0][j][4] = 0.0002 * Pd  # Load hash (aggregated demand signature)

    count_batch += 1

# Step 6: Normalization -------------------------------------------------------------------------------------------------
node_limits = pd.DataFrame(node_limits).T

V_norm = np.max(pd.concat([node_limits['V_min'], (
node_limits['V_max'])]))  # Structure here is not efficient. Is kept this way in case we want to change norms
P_norm = np.max(pd.concat([node_limits['P_min'], (node_limits['P_max'])]))
Q_norm = np.max(pd.concat([node_limits['Q_min'], (node_limits['Q_max'])]))
S_norm = np.max(edge_limits[0])
norm_coeffs = {'P_norm': P_norm, 'Q_norm': Q_norm, 'V_norm': 11, 'Theta_norm': 360., 'Base_MVA': Base_MVA}

for i in range(len(node_inputs)):
    for j in range(total_nb_nodes):
        node_inputs[i][0][j][0] = node_inputs[i][0][j][0] / norm_coeffs['P_norm']
        node_inputs[i][0][j][1] = node_inputs[i][0][j][1] / norm_coeffs['Q_norm']
        node_optimum[i][0][j][0] = node_optimum[i][0][j][0] / norm_coeffs['P_norm']
        node_optimum[i][0][j][1] = node_optimum[i][0][j][1] / norm_coeffs['Q_norm']
        node_optimum[i][0][j][2] = node_optimum[i][0][j][2] / norm_coeffs['V_norm']
        node_optimum[i][0][j][3] = node_optimum[i][0][j][3] / norm_coeffs['Theta_norm']

edge_index = torch.tensor(edge_index, dtype=torch.long)

# Node type mapping according to paper Table 3 (Section 2.5, page 9)
# Paper definition:
#   ti = 0 -> Load nodes (ND)
#   ti = 1 -> Generator nodes (NG)
#   ti = 2 -> Connection/Transmission nodes (NL)
node_types = []
for i in range(total_nb_nodes):
    original_type = nodes[i]['type']
    if original_type == 5:  # Generator nodes
        mapped_type = 1  # Paper Table 3: Generator = 1
    elif original_type == 1:  # Load nodes
        mapped_type = 0  # Paper Table 3: Load = 0
    else:  # Line/Connection/Other nodes
        mapped_type = 2  # Paper Table 3: Connection = 2
    node_types.append(mapped_type)
node_types = torch.tensor(node_types, dtype=torch.long).to(device)

# Step 7: Distribution of points into leaders --------------------------------------------------------------------------

data_list = []

X = {'X': node_inputs}
Y = {'Y': node_optimum}
for i in range(len(X['X'])):
    N = torch.tensor(X['X'][i][0], dtype=torch.float, device=device)
    Y_o = torch.tensor(Y['Y'][i][0], dtype=torch.float, device=device)
    data = Data(
        x=N.to(device),
        edge_index=edge_index.to(device),
        edge_attr=None,
        y=Y_o.to(device),
        node_type=node_types.clone().to(device)
    )  # Since all of the rest of the information is shared between all points, no reason to include it in each data point
    data_list.append(data)  # we removed the .to(device) from the edge_index

a, b, c = split
train_set, val_set, test_set = torch.utils.data.random_split(data_list, [a, b, c])
if len(node_inputs) >= 10:
    print('case')
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)

if len(node_inputs) < 10:
    print('Train, Validation and Test set will be the same')
    train_loader = DataLoader(data_list, batch_size=batch, shuffle=False)
    val_loader = DataLoader(data_list, batch_size=batch, shuffle=False)
    test_loader = DataLoader(data_list, batch_size=1, shuffle=False)

nodes = pd.DataFrame(nodes)
edge_limits = pd.DataFrame(edge_limits)
node_chars = torch.tensor(nodes.values).to(device)
static_costs_index = torch.tensor([3, 6]).to(device)
static_costs = torch.index_select(node_chars, dim=0, index=static_costs_index).to(torch.float32).to(device).sum()

characteristics = {'node_nb': nb_nodes, 'total_node_nb': total_nb_nodes, 'nodes': nodes, 'node_limits': node_limits,
                   'edge_limits': edge_limits, 'actual_edges': actual_edges, 'G': G, 'B': B, 'norm_coeffs': norm_coeffs,
                   'Ref_node': ref_node, 'gen_index': gen_index, 'gen_groups': gen_groups, 'gen_to_bus': gen_to_bus,
                   'edge_to_bus': edge_to_bus, 'static_costs': static_costs}
characteristics['static_cost'] = mpc['gencost'].T[6].sum()

print('Norm coefficients: ' + str(norm_coeffs))
print('Saving {}_{}_{}'.format(case, points, batch))
print('Avg function inferrence takes: %f' % (np.mean(np.array(avg_inference_time))))
torch.save(train_loader, "Input_Data/train_loader_{}_{}_{}.pt".format(case, points, batch))
torch.save(val_loader, "Input_Data/val_loader_{}_{}_{}.pt".format(case, points, batch))
torch.save(test_loader, "Input_Data/test_loader_{}_{}_{}.pt".format(case, points, batch))

with open('Input_Data/characteristics_{}.pkl'.format(case), 'wb') as f:
    pickle.dump(characteristics, f)

