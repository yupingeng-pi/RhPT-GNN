import pandapower as pp
import pandapower.networks as pn
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import torch
from decimal import Decimal
from time import time

from utils import *
import pickle

device = "cpu"

"""
Main Post-Processing script. This mirrors closely how the testing function of the TrainValTest.py script.
It works in the following way:
    1. Loads the model within the library of results.
    2. Opens a test data loader. The Preprocessing script can create special testing sets with batch size 1 (which is better for analysis).
    3. Computes all of the relevant losses (exactly the same code as in test_epoch_regr_pinn() of the TrainValTest.py file)
    4. Gathers the losses and averages them
    5. Prints out the relevant information
"""


# Main User Defined Inputs --------------------------------------------------------------------------------------------------
#case = "case9"
#case = "case30"
#case = "case24_ieee_rts"
case = "case118"
points = 20
batch_size = 1

n = 4 #number of decimals for results truncation


model_path = "Results\paper_results\M=2+\case118_clean\model.pt"
#model_path = "Results\paper_results\M=2+\case9_M=500\model.pt"
test_loader = torch.load("Input_Data/test_loader_{}_{}_{}.pt".format(case, points, batch_size))
#test_loader = torch.load("Input_Data/test_loader_{}_test_N-1_{}_{}.pt".format(case, points, batch_size))

# ---------------------------------------------------------------------------------------------------------------------------


def truncate_to_n_decimal_places(tensor, n):
    """Quick function to truncate to the nth decimal. It exclusively works on torch tensors."""
    tensor = tensor.detach().numpy()
    return np.floor(tensor* 10**n) / 10**n


with open('Input_Data/characteristics_{}.pkl'.format(case), 'rb') as f:
#with open('Input_Data/characteristics_{}_test_N-1.pkl'.format(case), 'rb') as f:
            characteristics = pickle.load(f)

model = torch.load(model_path)
model = model.to(device)
model.eval()

nb_nodes = characteristics['node_nb']
edge_from_bus = characteristics["edge_to_bus"][0][0]
edge_to_bus = characteristics["edge_to_bus"][0][1]
gen_to_bus = characteristics["gen_to_bus"]
edge_nb = len(characteristics['actual_edges'])
base_MVA = characteristics['norm_coeffs']['Base_MVA']
edge_limits = characteristics['edge_limits']
node_limits = characteristics['node_limits']
gen_index = torch.tensor(characteristics['gen_index'])
G = torch.tensor(characteristics['G']).unsqueeze(1)
B = torch.tensor(characteristics['B']).unsqueeze(1)
gen_nb = len(characteristics['gen_index'])
node_chars = torch.tensor(characteristics['nodes'].values)
total_nb_nodes = characteristics['total_node_nb']

eq_losses = []
eq_losses_true = []
flow_losses = []
flow_losses_true = []
gen_ineq_losses = []
gen_ineq_losses_true = []
node_ineq_losses = []
node_ineq_losses_true = []


cost_differences = []
true_costs = []

avg_inference_time = []
j = 0
for data in test_loader:
    j+=1
    norm_coeffs = characteristics['norm_coeffs']
    true = data.y
    t0 = time()
    pred = model(data)
    t1 = time()
    avg_inference_time.append((t1 - t0))

    for i in range(characteristics['total_node_nb']*batch_size):
            if i%characteristics['total_node_nb'] not in characteristics['gen_index']:
                pred.T[0][i] = 0
                pred.T[1][i] = 0
            if i%characteristics['total_node_nb'] > characteristics['node_nb']:
                pred.T[2][i] = 0
                pred.T[3][i] = 0
            if i%characteristics['total_node_nb'] == characteristics['Ref_node']:
                pred.T[3][i] = 0
        
        #print(i)
        #if n == 0:
    
    pd = data.x[:,0]*norm_coeffs['P_norm']
    qd = data.x[:,1]*norm_coeffs['Q_norm']
    pg = pred.T[0]*norm_coeffs['P_norm']
    qg = pred.T[1]*norm_coeffs['Q_norm']
    V = pred.T[2,:]*norm_coeffs['V_norm']
    Theta = pred.T[3,:]*np.pi/180*norm_coeffs['Theta_norm']

    #print()    

    Theta = Theta.view(batch_size, total_nb_nodes).T[:nb_nodes]
    V = V.view(batch_size, total_nb_nodes).T[:nb_nodes]
    pg = pg.view(batch_size, total_nb_nodes).T
    qg = qg.view(batch_size, total_nb_nodes).T
    if characteristics['Ref_node'] > 0:
        Theta[characteristics['Ref_node'],:] = 0

    pd = pd.view(batch_size, total_nb_nodes).T[:nb_nodes]
    qd = qd.view(batch_size, total_nb_nodes).T[:nb_nodes]

    static_costs = torch.full((1,batch_size), characteristics['static_costs'].sum()).squeeze(0)

    eq_loss , flow_loss  = compute_eq_loss(pg, qg, pd, qd, V, Theta, edge_from_bus, edge_to_bus, gen_to_bus, nb_nodes, edge_nb, base_MVA, edge_limits, G, B, batch_size, device) #We probably want to change this to get a better spread
    if j == 1:
          print(eq_loss)
    gen_ineq_loss, node_ineq_loss = compute_ineq_losses(pg,qg,V, node_limits, gen_index, nb_nodes, gen_nb, batch_size, device)

    cost_loss = compute_total_cost(pg,qg, gen_index, node_chars, batch_size, device, static_costs)
    #p_f, q_f = compute_flows(V, Theta, edge_from_bus, edge_to_bus, base_MVA, edge_limits, G, B, batch_size, device)

    true_pg = true.T[0]*norm_coeffs['P_norm']
    true_qg = true.T[1]*norm_coeffs['Q_norm']
    true_V = true.T[2]*norm_coeffs['V_norm']
    true_Theta = true.T[3]*np.pi/180*norm_coeffs['Theta_norm']
        
    true_Theta = true_Theta.view(batch_size, total_nb_nodes).T[:nb_nodes]
    true_V = true_V.view(batch_size, total_nb_nodes).T[:nb_nodes]
    true_pg = true_pg.view(batch_size, total_nb_nodes).T
    true_qg = true_qg.view(batch_size, total_nb_nodes).T
    if characteristics['Ref_node'] > 0:
        true_Theta[characteristics['Ref_node'],:] = 0
        
    eq_loss_true , flow_loss_true = compute_eq_loss(true_pg, true_qg, pd, qd, true_V, true_Theta, edge_from_bus, edge_to_bus, gen_to_bus, nb_nodes, edge_nb, base_MVA, edge_limits, G, B, batch_size, device)
    
    gen_ineq_loss_true, node_ineq_loss_true = compute_ineq_losses(true_pg, true_qg, true_V, node_limits, gen_index, nb_nodes, gen_nb, batch_size, device)

    true_cost = compute_total_cost(true_pg,true_qg, gen_index, node_chars, batch_size, device, static_costs)

    eq_loss = truncate_to_n_decimal_places(torch.abs(eq_loss).sum(), n)
    eq_losses.append(eq_loss)

    eq_loss_true = truncate_to_n_decimal_places(torch.abs(eq_loss_true).sum(), n)
    eq_losses_true.append(eq_loss_true)

    flow_loss = truncate_to_n_decimal_places(torch.abs(flow_loss).sum(), n)
    flow_losses.append(flow_loss)

    flow_loss_true = truncate_to_n_decimal_places(torch.abs(flow_loss_true).sum(), n)
    flow_losses_true.append(flow_loss_true)

    gen_ineq_loss = truncate_to_n_decimal_places(torch.abs(gen_ineq_loss).sum(), n)
    gen_ineq_losses.append(gen_ineq_loss)

    gen_ineq_loss_true = truncate_to_n_decimal_places(torch.abs(gen_ineq_loss_true).sum(), n)
    gen_ineq_losses_true.append(gen_ineq_loss_true)

    node_ineq_loss = truncate_to_n_decimal_places(torch.abs(node_ineq_loss).sum(), n)
    node_ineq_losses.append(node_ineq_loss)

    node_ineq_loss_true = truncate_to_n_decimal_places(torch.abs(node_ineq_loss_true).sum(), n)
    node_ineq_losses_true.append(node_ineq_loss_true)

    cost_difference = truncate_to_n_decimal_places((cost_loss - true_cost), n)
    cost_differences.append(cost_difference)

    true_costs.append(true_cost)

#"""
print('Raw numbers:')
print('Equality losses: ')
print(eq_losses)
print(eq_losses_true)
print('Flow Losses: ')
print(flow_losses)
print(flow_losses_true)
print('Gen Ineq Losses: ')
print(gen_ineq_losses)
print(gen_ineq_losses_true)
print('Node Ineq Losses: ')
print(node_ineq_losses)
print(node_ineq_losses_true)
print('Cost differences:')
print(cost_differences)
#"""

#"""
print()
print('Averages:')
print(np.mean(np.array(eq_losses)))
print(np.mean(np.array(eq_losses_true)))
avg_cost_difference = np.mean(np.array(cost_differences))
print(avg_cost_difference)
avg_true_cost = np.mean(np.array(true_costs))
print(avg_true_cost)
print(100*avg_cost_difference/avg_true_cost)

print("Average Inference times:")
print(np.mean(np.array(avg_inference_time)))
# case24_ieee_rts = 0.003418755531311035
# case9 = 0.0021962642669677733
# case30 = 0.0031243071836583756
#"""
#case30_500_20 --> 416