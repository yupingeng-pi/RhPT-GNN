from torch import square, relu, zeros, tensor, sin, cos, sqrt, float
import torch
from torch.nn.functional import relu, l1_loss
#import torch_geometric
#import numpy as np
#import matplotlib.pyplot as plt
#import time
import pandas as pds
import numpy as np


def compute_eq_loss(pg,qg,pd,qd, V, Theta, edge_from_bus, edge_to_bus, gen_to_bus, nb_nodes, edge_nb, base_MVA, edge_limits, G, B, batch_size, device):    
    """
    Function to compute the equality losses and the flow losses. Intakes the grid characteristics and inputs and outputs the batch-wise averaged losses
    """

    # Make sure everything is sent to the device
    pg.to(device)
    qg.to(device)
    pd.to(device)
    qd.to(device)
    V.to(device)
    Theta.to(device)
    edge_from_bus.to(device)
    edge_to_bus.to(device)
    gen_to_bus.to(device)
    G = G.to(device)
    B = B.to(device)

    #Transform everything to the proper matrix form
    Theta_diff = edge_from_bus @ Theta - edge_to_bus @ Theta
    V_from = edge_from_bus @ V
    V_to = edge_to_bus @ V

    tap = torch.tensor(edge_limits[2]).unsqueeze(1).to(device)
    shunts = torch.tensor(edge_limits[1]).unsqueeze(1).to(device)
    S_ij = torch.tensor(edge_limits[0]).unsqueeze(1).repeat(1,batch_size).to(device)

    #Power Flows
    p_f = base_MVA * (
    square(V_from) * G * tap
    - V_from * V_to * G * torch.cos(Theta_diff)
    - V_from * V_to * B * torch.sin(Theta_diff)
    ).to(device)

    q_f = base_MVA * (
    -square(V_from) * (B + shunts / 2) * tap
    + V_from * V_to * B * torch.cos(Theta_diff)
    - V_from * V_to * G * torch.sin(Theta_diff)
    ).to(device)
    
    flow_loss = torch.relu(torch.square(p_f) + torch.square(q_f) - torch.square(S_ij)).to(device)
    flow_loss = flow_loss.mean(dim=1).to(device) #Batch-wise average of tge loss

    #Equalities
    Pg = (gen_to_bus @ pg).to(device)
    Qg = (gen_to_bus @ qg).to(device)

    res_pg = (edge_from_bus.T @ p_f.to(torch.float32)).to(device)
    res_qg = (edge_from_bus.T @ q_f.to(torch.float32)).to(device)

    res_pg = torch.abs(res_pg + pd - Pg).to(device)
    res_pg = res_pg.mean(dim=1).to(device)
    res_qg = torch.abs(res_qg + qd - Qg).to(device)
    res_qg = res_qg.mean(dim=1).to(device)

    eq_loss = torch.stack((res_pg,res_qg), dim = 0).to(device)

    return eq_loss, flow_loss

def compute_ineq_losses(pg, qg, V, node_limits, gen_index, nb_nodes, gen_nb, batch_size, device):
    """
    Function to compute the inequality losses.
    """
    #Inequalities

    P_max = torch.index_select(torch.tensor(node_limits['P_max']).to(device), dim=0, index = gen_index).unsqueeze(1).repeat(1,batch_size).to(device)
    P_min = torch.index_select(torch.tensor(node_limits['P_min']).to(device), dim=0, index = gen_index).unsqueeze(1).repeat(1,batch_size).to(device)
    Q_max = torch.index_select(torch.tensor(node_limits['Q_max']).to(device), dim=0, index = gen_index).unsqueeze(1).repeat(1,batch_size).to(device)
    Q_min = torch.index_select(torch.tensor(node_limits['Q_min']).to(device), dim=0, index = gen_index).unsqueeze(1).repeat(1,batch_size).to(device)

    gen_pg = torch.index_select(pg, dim=0, index = gen_index).to(device)
    gen_qg = torch.index_select(qg, dim=0, index = gen_index).to(device)

    P_max_loss = torch.relu(gen_pg - P_max).to(device)
    P_max_loss = P_max_loss.mean(dim=1).to(device)

    P_min_loss = torch.relu(P_min - gen_pg).to(device)
    P_min_loss = P_min_loss.mean(dim=1).to(device)

    Q_max_loss = torch.relu(gen_qg - Q_max).to(device)
    Q_max_loss = Q_max_loss.mean(dim=1).to(device)

    Q_min_loss = torch.relu(Q_min - gen_qg).to(device)
    Q_min_loss = Q_min_loss.mean(dim=1).to(device)
    

    gen_ineq_loss = torch.stack((P_max_loss, P_min_loss, Q_max_loss, Q_min_loss), dim = 0).to(device)

    V_max = torch.tensor(node_limits['V_max'][:nb_nodes]).to(device).unsqueeze(1).repeat(1,batch_size).to(device)
    V_min = torch.tensor(node_limits['V_min'][:nb_nodes]).to(device).unsqueeze(1).repeat(1,batch_size).to(device)

    V_max_loss = (torch.relu(V - V_max)*10).to(device) # !!! Importantly there is a factor here. it ensures that voltage magnitudes are respected. Not sure if still relevant
    V_max_loss = V_max_loss.mean(dim=1).to(device)

    V_min_loss = (torch.relu(V_min - V)*10).to(device)
    V_min_loss = V_min_loss.mean(dim=1).to(device)

    node_ineq_loss = torch.stack((V_max_loss, V_min_loss), dim = 0).to(device)
    
    return gen_ineq_loss, node_ineq_loss

def compute_total_cost(pg,qg, gen_index, node_chars, batch_size, device, static_costs):
    """
    Function to compute total costs. Used for both the model output and the solution output.
    """
    #Costs
    node_chars = torch.index_select(node_chars, dim = 1, index = gen_index).to(device)
    cost_index = torch.tensor([1,2,4,5]).to(device) #This selects cp_1, cp_2, cq_1 and cq_2

    gen_pg = torch.index_select(pg, dim=0, index = gen_index).to(device)
    gen_qg = torch.index_select(qg, dim=0, index = gen_index).to(device)

    costs = torch.index_select(node_chars, dim = 0, index = cost_index).to(torch.float32).to(device)

    total_costs = (costs[0] @ torch.square(gen_pg) + costs[1] @ gen_pg + 3*costs[2] @ torch.square(gen_qg) + 3*costs[3] @ gen_qg).to(device)

    return (total_costs + static_costs).mean(dim=0).to(device) # Averaged over batch


def penalty_multipliers_init(penalty_multipliers):
    '''
    Initialize the penalty_multipler dictionnary
    '''
    mun0, muf0, muh0, betaf, betah = penalty_multipliers
    penalty_multipliers = {}
    penalty_multipliers['mun'] = mun0
    penalty_multipliers['muf'] = muf0
    penalty_multipliers['muh'] = muh0
    penalty_multipliers['betaf'] = betaf
    penalty_multipliers['betah'] = betah
    return penalty_multipliers


def compute_flows(V, Theta, edge_from_bus, edge_to_bus, base_MVA, edge_limits, G, B, batch_size, device):    
    """
    Quick function to compute power flows. Usually used for debugging purposes.
    """
    Theta_diff = edge_from_bus @ Theta - edge_to_bus @ Theta
    V_from = edge_from_bus @ V
    V_to = edge_to_bus @ V
    tap = torch.tensor(edge_limits[2]).unsqueeze(1).to(device)
    shunts = torch.tensor(edge_limits[1]).unsqueeze(1).to(device)
    S_ij = torch.tensor(edge_limits[0]).unsqueeze(1).repeat(1,batch_size).to(device)

    p_f = base_MVA * (
    square(V_from) * G * tap
    - V_from * V_to * G * torch.cos(Theta_diff)
    - V_from * V_to * B * torch.sin(Theta_diff)
    ).to(device)

    q_f = base_MVA * (
    -square(V_from) * (B + shunts / 2) * tap
    + V_from * V_to * B * torch.cos(Theta_diff)
    - V_from * V_to * G * torch.sin(Theta_diff)
    ).to(device)
    
    return p_f,q_f