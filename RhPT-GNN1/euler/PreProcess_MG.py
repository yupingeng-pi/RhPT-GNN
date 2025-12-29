# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import copy
import pandas as pd
import pickle
from matpower import start_instance

# from pypower.api import case9Q, ppoption, runpf, printpf, makeYbus, case14, case30Q, case9, opf, case118, runopf

distrib = 'uniform'
m = start_instance()


class OPFGraph:
    def __init__(self, nodes, node_limits, edges, G, B, edge_index, Base_MVA):

        # Nodes
        self.nodes = pd.DataFrame(
            nodes)  # Dictionnary of nodes with information: [bus type, P_d, Q_d, c_1, c_2, c_3] -> Original P_d and Q_d
        self.node_limits = pd.DataFrame(
            node_limits).T  # Dictionnary of nodes with: [P_max, P_min, Q_max, Q_min, V_max, V_min]
        self.node_inputs = []  # This is a list of all inputs created by a custom distribution
        self.node_results = []  # Results from the NN
        self.node_optimum = []  # Validation data
        self.node_nb = len(list(self.nodes.keys()))
        self.ref_node = 0
        self.gen_index = []
        self.gen_groups = []
        self.original_nodes = list(self.nodes.columns)
        # print(self.original_nodes)
        # print(self.nodes)
        # !!! We don't enforce reference node so that probably !!!

        # Edges
        # self.edges = edges #Dictionnary with the edges in COO format: {node1: {node2: [S_ij_max, theta_max, theta_min]}} --> Not a good way
        self.edge_index = edge_index

        # Normalisation coefficients
        self.norm_coeffs = {}
        self.Base_MVA = Base_MVA

        # Get the simplified actual edges
        actual_edges = []
        actual_edge_limits = []
        Gs = []
        Bs = []
        for n in range(len(edge_index[0])):
            i = int(edge_index[0][n])
            j = int(edge_index[1][n])
            if i != j:
                if ([i, j] not in actual_edges):
                    actual_edges.append([i, j])
                    actual_edge_limits.append(
                        [edges[i][j]['Sij_max'], edges[i][j]['line_shunts'], edges[i][j]['theta_max'],
                         edges[i][j]['theta_min']])
                    Gs.append(-G[n])  # Importantly the makeYbus function gives us the wrong sign
                    Bs.append(-B[n])
        self.actual_edges = actual_edges
        # print(actual_edges)
        self.actual_edge_limits = pd.DataFrame(actual_edge_limits)
        self.G = Gs
        self.B = Bs

        for i in range(self.node_nb):
            if self.nodes[i]['type'] >= 2:
                self.gen_index.append(i)

    def SplitNode(self, gen_to_split, gen_characteristics):
        gen_group = [gen_to_split]
        self.nodes[gen_to_split]['type'] += 3  # 5 means dummy node with no demand, 6 means with demand
        self.nodes[gen_to_split]['cp_1'] = 0
        self.nodes[gen_to_split]['cp_2'] = 0
        self.nodes[gen_to_split]['cp_3'] = 0
        self.nodes[gen_to_split]['cq_1'] = 0
        self.nodes[gen_to_split]['cq_2'] = 0
        self.nodes[gen_to_split]['cq_3'] = 0
        self.gen_index.pop(gen_to_split)
        # print(self.actual_edges)
        for i in range(len(gen_characteristics)):
            # print(i)
            new_gen = pd.Series({'type': 4.0, 'P_d': 0.0, 'Q_d': 0.0,
                                 'cp_1': gen_characteristics['costs'][i][0],
                                 'cp_2': gen_characteristics['costs'][i][1],
                                 'cp_3': gen_characteristics['costs'][i][2],
                                 'cq_1': gen_characteristics['costs'][i][3],
                                 'cq_2': gen_characteristics['costs'][i][4],
                                 'cq_3': gen_characteristics['costs'][i][5]})
            self.nodes[len(self.nodes) + i] = new_gen
            gen_group.append(len(self.nodes) + i)
            new_limits = pd.Series({'P_max': gen_characteristics['limits'][i][0],
                                    'P_min': gen_characteristics['limits'][i][1],
                                    'Q_max': gen_characteristics['limits'][i][2],
                                    'Q_min': gen_characteristics['limits'][i][3],
                                    'V_max': gen_characteristics['limits'][i][4],
                                    'V_min': gen_characteristics['limits'][i][5]})
            self.node_limits.loc[len(self.nodes) + i] = new_limits
            self.gen_index.append(len(self.nodes) + i)
            self.node_nb += 1
            # We add the edge between the dummy node and the split generator nodes. We don't add them to the actual_edges however, as these edges don't matter in the nodal flow equations
            self.edge_index[0].append(gen_to_split)
            self.edge_index[1].append(len(self.nodes) + i)
            self.edge_index[1].append(gen_to_split)
            self.edge_index[0].append(len(self.nodes) + i)
        self.gen_groups.append(gen_group)
        # Since we don't solve anything from here (since matpower doesn't take this kind of input, we have to reproduce it ourselved)
        sample = np.zeros((self.node_nb, 4))
        for i in range(self.node_nb):
            sample[i][0] = self.nodes[i]['P_d']
            sample[i][1] = self.nodes[i]['Q_d']
            sample[i][2] = self.nodes[i]['type'] / 6
            sample[i][3] = i / self.node_nb
        self.node_inputs.append([sample])  # We assume there's only one input, it's probably going to bug otherwise
        self.node_optimum.append([np.zeros((self.node_nb, 4))])  # Essentially we say the solution is all 0

    def GeneratePoints(self, bounds, samples):
        P_distributions = []
        Q_distributions = []
        for i in range(self.node_nb):  # This is where we can change the distribution of inputs
            P_distrib = np.random.uniform((1 - bounds) * self.nodes[i]['P_d'], (1 + bounds) * self.nodes[i]['P_d'],
                                          size=samples)
            P_distributions.append(P_distrib)
            Q_distrib = np.random.uniform((1 - bounds) * self.nodes[i]['Q_d'], (1 + bounds) * self.nodes[i]['Q_d'],
                                          size=samples)
            Q_distributions.append(Q_distrib)
        # print(P_distributions)
        for j in range(samples):
            sample = np.zeros((self.node_nb, 4))
            for i in range(self.node_nb):
                sample[i][0] = P_distributions[i][j]
                sample[i][1] = Q_distributions[i][j]
                sample[i][2] = self.nodes[i]['type'] / 3
                sample[i][3] = i / self.node_nb
            self.node_inputs.append([sample])
        # print(self.node_inputs)

    def SolveInputs(self, mpc):  # Has to be changed in order to make it more general eventually
        # print(self.node_inputs)
        for sample in self.node_inputs:
            # print(sample)
            for i in range(self.node_nb):  # We change the values of the input with the new Pd and Qd
                mpc['bus'][i][2] = sample[0][i][0]
                mpc['bus'][i][3] = sample[0][i][1]
            mpopt = m.mpoption('verbose', 2, 'opf.ac.solver', 'MIPS',
                               'mips.max_it', 150,
                               'mips.gradtol', 1e-8,
                               'mips.comptol', 1e-8,
                               'mips.costtol', 1e-8)
            r = m.runopf(mpc, mpopt, nout='max_nout')  # Currently no exception if this fails
            r = {'baseMVA': r[0], 'bus': r[1], 'gen': r[2], 'gencost': r[3], 'branch': r[4]}

            # for i in range(len(mpc['bus'])):
            #    r['bus'][i][0]+= -1
            # for i in range(len(mpc['branch'])):
            #    r['branch'][i][0]+= -1
            #    r['branch'][i][1]+= -1
            # for i in range(len(mpc['gen'])):
            #    r['gen'][i][0] += -1
            #    r['gencost'][i][0] += -1

            sample_solution = np.zeros((self.node_nb, 4))
            gen_count = 0
            for i in range(self.node_nb):
                if self.nodes[i]['type'] == 2 or self.nodes[i]['type'] == 3:
                    sample_solution[i][0] = r['gen'][gen_count][1]
                    sample_solution[i][1] = r['gen'][gen_count][2]
                    gen_count += 1
                else:
                    sample_solution[i][0] = 0
                    sample_solution[i][1] = 0
                sample_solution[i][2] = r['bus'][i][7]
                sample_solution[i][3] = r['bus'][i][8]
            self.node_optimum.append([sample_solution])
        # print(self.node_optimum)
        for i in range(self.node_nb):
            if self.node_optimum[0][0][i][3] == 0:
                self.ref_node = i
        # print(self.node_optimum)
        return None

    def Edge_index_to_tensor(self):
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)

    def Repackage(self, list):
        temp_list = []
        for nb in list:
            temp_list.append([nb])
        return np.array(temp_list)

    def L2Norm(self, array):
        sum = 0
        for nb in array:
            sum += nb * nb
        sum = np.sqrt(sum)
        array = array / sum
        return sum, array

    def MaxNorm(self, array):
        max = np.max(array)
        return max, array / max

    def Normalize(self):  # Haven't normalized everything yet TODO: Change the max norm to something more centered
        device = 'cpu'
        # We don't actually need to normalize these, only the inputs and outputs
        # G_norm, Gs = self.L2Norm(self.G)
        # G_norm, self.G = self.L2Norm(self.G)
        # B_norm, Bs = self.L2Norm(self.B)
        # B_norm, self.B = self.L2Norm(self.B)

        """
        P_max = np.max(self.node_limits['P_max'])
        P_min = np.min(self.node_limits['P_min'])
        Q_max = np.max(self.node_limits['Q_max'])
        Q_min = np.min(self.node_limits['Q_min'])
        self.norm_coeffs = {'P_max': P_max, 'P_min': P_min, 'Q_max': Q_max,'Q_min': Q_min, 'Theta_norm': 36., 'Base_MVA': self.Base_MVA}
        """

        # """
        V_norm = np.max(pd.concat([self.node_limits['V_min'], (self.node_limits[
            'V_max'])]))  # Structure here is not efficient. Is kept this way in case we want to change norms
        P_norm = np.max(pd.concat([self.node_limits['P_min'], (self.node_limits['P_max'])]))
        Q_norm = np.max(pd.concat([self.node_limits['Q_min'], (self.node_limits['Q_max'])]))
        S_norm = np.max(self.actual_edge_limits[0])
        self.norm_coeffs = {'V_norm': 11, 'P_norm': P_norm, 'Q_norm': Q_norm, 'S_norm': S_norm, 'Theta_norm': 360.,
                            'Base_MVA': self.Base_MVA}
        # """
        print(P_norm)
        print(Q_norm)
        print(self.node_inputs[0][0])
        # We then normalize the inputs and outputs
        for i in range(len(self.node_inputs)):
            for j in range(self.node_nb):
                """
                self.node_inputs[i][0][j][0] = (self.node_inputs[i][0][j][0] - P_min)/(P_max - P_min)
                self.node_inputs[i][0][j][1] = (self.node_inputs[i][0][j][1] - Q_min)/(Q_max - Q_min)
                self.node_optimum[i][0][j][0] = (self.node_optimum[i][0][j][0] - P_min)/(P_max - P_min)
                self.node_optimum[i][0][j][1] = (self.node_optimum[i][0][j][1] - Q_min)/(Q_max - Q_min)
                self.node_optimum[i][0][j][2] = (self.node_optimum[i][0][j][2] - 0.9)/(0.2)# * 10
                self.node_optimum[i][0][j][3] = (self.node_optimum[i][0][j][3] + 360)/(720)# * 10
                #"""

                # """
                self.node_inputs[i][0][j][0] = self.node_inputs[i][0][j][0] / self.norm_coeffs['P_norm']
                self.node_inputs[i][0][j][1] = self.node_inputs[i][0][j][1] / self.norm_coeffs['Q_norm']
                self.node_optimum[i][0][j][0] = self.node_optimum[i][0][j][0] / self.norm_coeffs['P_norm']
                self.node_optimum[i][0][j][1] = self.node_optimum[i][0][j][1] / self.norm_coeffs['Q_norm']
                self.node_optimum[i][0][j][2] = self.node_optimum[i][0][j][2] / self.norm_coeffs['V_norm']
                self.node_optimum[i][0][j][3] = self.node_optimum[i][0][j][3] / self.norm_coeffs['Theta_norm']
                # """
        # print(self.node_optimum)
        return None

    def GetCharacteristics(self):
        return {'node_nb': self.node_nb, 'total_node_nb': self.node_nb, 'nodes': self.nodes,
                'node_limits': self.node_limits, 'edge_limits': self.actual_edge_limits, 'edge_index': self.edge_index,
                'actual_edges': self.actual_edges, 'G': self.G, 'B': self.B, 'norm_coeffs': self.norm_coeffs,
                'Ref_node': self.ref_node,
                'gen_index': self.gen_index}  # ,'gen_groups': self.gen_groups, 'original_nodes': self.original_nodes}

    def DataSplit(self, split, batch_size):
        data_list = []
        device = 'cpu'
        X = {'X': self.node_inputs}
        Y = {'Y': self.node_optimum}
        # print(self.node_inputs)
        # print(X['X'][3][0])
        # print(Y['Y'])
        # print(self.actual_edges)
        # We reformat the acutal edges to fit the data input shape
        # edge_x = []
        # edge_y = []
        # print(actual_edges)
        # for j in range(len(self.actual_edges)):
        #    edge_x.append(self.actual_edges[j][0])
        #    edge_y.append(self.actual_edges[j][1])
        # actual_edges = [edge_x, edge_y]
        # actual_edges = torch.tensor(actual_edges, dtype=torch.long)
        # print(self.edge_index)
        # print(actual_edges)
        # print(X['X'])
        print(self.edge_index)
        # """
        for i in range(len(X['X'])):
            N = torch.tensor(X['X'][i][0], dtype=torch.float, device=device)
            Y_o = torch.tensor(Y['Y'][i][0], dtype=torch.float, device=device)
            data = Data(x=N, edge_index=self.edge_index, edge_attr=None, y=Y_o).to(
                device)  # Since all of the rest of the information is shared between all points, no reason to include it in each data point
            # data = Data(x=N, y=Y_o).to(device)
            data_list.append(data)  # we removed the .to(device) from the edge_index
        if len(X['X']) > 10:
            train_set, val_set, test_set = torch.utils.data.random_split(data_list, split)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        else:
            print('Train, Validation and Test set will be the same')
            train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        # print(data_list)
        # """
        # return X,Y
        return train_loader, val_loader, test_loader  # We don't create a small batch

    def Snapshot(self):
        return None


# mpc = case9Q()
# print(mpc['branch'])
# print(mpc['bus'])
# mpc['bus'][4][2] += 10
# print(mpc['bus'][4][2])
# [Ybus, Yf, Yt] = makeYbus(mpc['baseMVA'],mpc['bus'],mpc['branch'])
# print(Ybus.data[1])
# for i in range(9):
#    mpc['bus'][i][2] = 0.9*mpc['bus'][i][2]
#    mpc['bus'][i][3] = 0.9*mpc['bus'][i][3]

# ppopt = ppoption(PF_ALG=2)
# r = runpf(mpc, ppopt)
# print(r)
# print(mpc['gencost'][1][5])

def build_OPFGraph(case, points, spread, split, batch_size):
    Q_case = False
    if case[-1] == 'Q':
        Q_case = True
        # print('Q case')
    mpc = m.loadcase(case)

    # mpc['gencost'][0][4] = 0
    # mpc['gencost'][0][5] = 100
    # mpc['gencost'][1][4] = 0
    # mpc['gencost'][1][5] = 10 #TODO: try with a greater value of cost. It might be too low at 1
    # mpc['gencost'][2][4] = 0
    # mpc['gencost'][2][5] = 20

    # mpc['gencost'][3][4] = 0
    # mpc['gencost'][3][5] = 0.5
    # mpc['gencost'][4][4] = 0
    # mpc['gencost'][4][5] = 0.1 #TODO: try with a greater value of cost. It might be too low at 1
    # mpc['gencost'][5][4] = 0
    # mpc['gencost'][5][5] = 0.2

    # mpc['bus'][8][2] = 25

    # for i in range(len(mpc['bus'])):
    #    mpc['bus'][i][0]+= -1
    # for i in range(len(mpc['branch'])):
    #    mpc['branch'][i][0]+= -1
    #    mpc['branch'][i][1]+= -1
    # for i in range(len(mpc['gen'])):
    #    mpc['gen'][i][0] += -1
    #    mpc['gencost'][i][0] += -1

    nb_nodes = len(mpc['bus'])
    nodes = {}
    node_limits = {}
    # node_optimum = {}

    # print(mpc['gen'][0][:])
    # print(mpc['bus'])
    # print(mpc['gen'])
    gen_count = 0
    # mpc['bus'][4][2] = 20
    # mpc['bus'][6][2] = 30
    # mpc['bus'][8][2] = 25
    # mpc['bus'][4][3] = 3
    # mpc['bus'][6][3] = 3.5
    # mpc['bus'][8][3] = 5
    # print(mpc['gencost'])
    nb_gens = len(mpc['gen'])
    # print(mpc['bus'])
    for i in range(nb_nodes):
        type = 0  # This corresponds to buses with no loads or generators
        if mpc['bus'][i][2] > 0 or mpc['bus'][i][3] > 0:
            type = 1  # Load buses
        for j in range(nb_gens):
            if i == mpc['gen'][j][0] - 1:
                if type == 1:
                    type = 3  # Bus with both a load and a generator
                else:
                    type = 2  # Generator buses
        # print(i)
        # print('Type:' + str(type))
        if type == 0:  # Empty
            nodes[i] = {'type': type, 'P_d': 0., 'Q_d': 0., 'cp_1': 0., 'cp_2': 0., 'cp_3': 0., 'cq_1': 0., 'cq_2': 0.,
                        'cq_3': 0.}
            node_limits[i] = {'P_max': 0., 'P_min': 0., 'Q_max': 0., 'Q_min': 0., 'V_max': mpc['bus'][i][11],
                              'V_min': mpc['bus'][i][12]}
            # node_limits[i] = {'P_max': 1e-6,'P_min': -1e-6,'Q_max': 1e-6,'Q_min': -1e-6,'V_max': mpc['bus'][i][11], 'V_min': mpc['bus'][i][12]}
        elif type == 1:  # Loads
            nodes[i] = {'type': type, 'P_d': mpc['bus'][i][2], 'Q_d': mpc['bus'][i][3], 'cp_1': 0., 'cp_2': 0.,
                        'cp_3': 0., 'cq_1': 0., 'cq_2': 0., 'cq_3': 0.}
            node_limits[i] = {'P_max': 0., 'P_min': 0., 'Q_max': 0., 'Q_min': 0., 'V_max': mpc['bus'][i][11],
                              'V_min': mpc['bus'][i][12]}
            # node_limits[i] = {'P_max': 1e-6,'P_min': -1e-6,'Q_max': 1e-6,'Q_min': -1e-6,'V_max': mpc['bus'][i][11], 'V_min': mpc['bus'][i][12]}
        elif type == 2:  # Generator
            if Q_case:
                nodes[i] = {'type': type, 'P_d': 0., 'Q_d': 0., 'cp_1': mpc['gencost'][gen_count][4],
                            'cp_2': mpc['gencost'][gen_count][5], 'cp_3': mpc['gencost'][gen_count][6],
                            'cq_1': mpc['gencost'][gen_count + nb_gens][4],
                            'cq_2': mpc['gencost'][gen_count + nb_gens][5],
                            'cq_3': mpc['gencost'][gen_count + nb_gens][6]}
            else:
                nodes[i] = {'type': type, 'P_d': 0., 'Q_d': 0., 'cp_1': mpc['gencost'][gen_count][4],
                            'cp_2': mpc['gencost'][gen_count][5], 'cp_3': mpc['gencost'][gen_count][6], 'cq_1': 0.,
                            'cq_2': 0., 'cq_3': 0.}
            node_limits[i] = {'P_max': mpc['gen'][gen_count][8], 'P_min': mpc['gen'][gen_count][9],
                              'Q_max': mpc['gen'][gen_count][3], 'Q_min': mpc['gen'][gen_count][4],
                              'V_max': mpc['bus'][i][11], 'V_min': mpc['bus'][i][12]}
            gen_count += 1
        else:  # Generator + Load
            if Q_case:
                nodes[i] = {'type': type, 'P_d': mpc['bus'][i][2], 'Q_d': mpc['bus'][i][2],
                            'cp_1': mpc['gencost'][gen_count][4], 'cp_2': mpc['gencost'][gen_count][5],
                            'cp_3': mpc['gencost'][gen_count][6], 'cq_1': mpc['gencost'][gen_count + nb_gens][4],
                            'cq_2': mpc['gencost'][gen_count + nb_gens][5],
                            'cq_3': mpc['gencost'][gen_count + nb_gens][6]}
            else:
                nodes[i] = {'type': type, 'P_d': mpc['bus'][i][2], 'Q_d': mpc['bus'][i][2],
                            'cp_1': mpc['gencost'][gen_count][4], 'cp_2': mpc['gencost'][gen_count][5],
                            'cp_3': mpc['gencost'][gen_count][6], 'cq_1': 0., 'cq_2': 0., 'cq_3': 0.}
            node_limits[i] = {'P_max': mpc['gen'][gen_count][8], 'P_min': mpc['gen'][gen_count][9],
                              'Q_max': mpc['gen'][gen_count][3], 'Q_min': mpc['gen'][gen_count][4],
                              'V_max': mpc['bus'][i][11], 'V_min': mpc['bus'][i][12]}
            gen_count += 1
    # print(node_limits)
    edges = {}
    edge_index = []
    for line in mpc['branch']:
        edges[int(line[0]) - 1] = {}
        edges[int(line[1]) - 1] = {}
        edge_index.append([int(line[0]) - 1, int(line[1]) - 1])
        edge_index.append([int(line[0]) - 1, int(line[0]) - 1])
    # print(edges)

    for line in mpc['branch']:
        edges[int(line[0]) - 1][int(line[1]) - 1] = {'Sij_max': line[5], 'line_shunts': line[4], 'theta_max': line[11],
                                                     'theta_min': line[
                                                         12]}  # This isn't correct in the original code. It isn't symmetric
        edges[int(line[1]) - 1][int(line[0]) - 1] = {'Sij_max': line[5], 'line_shunts': line[4], 'theta_max': line[11],
                                                     'theta_min': line[12]}
        # edges[int(line[0])][int(line[0])] = {'Sij_max': 0., 'theta_max': 0., 'theta_min': 0.} #A bit redundant but not that bad,
        # print(edges)
    Ybus = m.makeYbus(mpc['baseMVA'], mpc['bus'], mpc['branch'])
    edge_index = [list(Ybus.indices), sorted(Ybus.indices)]
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    G = np.real(Ybus).data
    B = np.imag(Ybus).data
    Base_MVA = mpc['baseMVA']

    graph = OPFGraph(nodes, node_limits, edges, G, B, edge_index, Base_MVA)
    # graph.SplitNode(1, {'costs':[[0.0,1.0,0.0,0.0,0.0,0.0],[0.0,100.0,0.0,0.0,0.0,0.0]], 'limits': [[300,10,300,-300,1.1,0.9],[300,10,300,-300,1.1,0.9]]})

    # graph.Edge_index_to_tensor()
    graph.GeneratePoints(spread, points)
    graph.SolveInputs(mpc)
    graph.Normalize()
    # characteristics = graph.GetCharacteristics()

    train_loader, val_loader, test_loader = graph.DataSplit(split, batch_size)
    characteristics = graph.GetCharacteristics()
    # print(graph.nodes)
    # print(graph.node_limits)
    # print(graph.node_inputs)
    # print(graph.edge_index)
    # print(characteristics)

    torch.save(train_loader, "Input_Data/train_loader_{}_{}_{}.pt".format(case, points, batch_size))
    torch.save(val_loader, "Input_Data/val_loader_{}_{}_{}.pt".format(case, points, batch_size))
    torch.save(test_loader, "Input_Data/test_loader_{}_{}_{}.pt".format(case, points, batch_size))
    with open('Input_Data/characteristics_{}.pkl'.format(case), 'wb') as f:
        pickle.dump(characteristics, f)

    # return characteristics, train_loader, val_loader, test_loader
    return None


build_OPFGraph(case="case9", points=1, spread=0.0, split=[0.8, 0.1, 0.1], batch_size=1)

# for i, data in enumerate(b): # Goes over each batch
#        batch_size = data.num_graphs
#        print(data)
#        print(batch_size)

# print(graph.G)
# print(graph.B)
# print()
# print(pd.DataFrame(graph.node_inputs[0][0].T[0]))
# print(graph.actual_edges)
# print(graph.actual_edge_limits)
# """
mpc = m.loadcase('case9')
mpc['gencost'][0][4] = 0
mpc['gencost'][0][5] = 30
mpc['gencost'][1][4] = 0
mpc['gencost'][1][5] = 1
mpc['gencost'][2][4] = 0
mpc['gencost'][2][5] = 20
# mpc['bus'][8][2] = 25
clone = copy.deepcopy(mpc)
for i in range(len(mpc['bus'])):
    mpc['bus'][i][0] += -1
for i in range(len(mpc['branch'])):
    mpc['branch'][i][0] += -1
    mpc['branch'][i][1] += -1
for i in range(len(mpc['gen'])):
    mpc['gen'][i][0] += -1
    mpc['gencost'][i][0] += -1
# [Ybus, Yf, Yt] = makeYbus(mpc['baseMVA'],mpc['bus'],mpc['branch'])
mpopt = m.mpoption('verbose', 2)
# output = m.runopf(clone, mpopt, nout='max_nout')
# output = {'baseMVA': output[0], 'bus': output[1], 'gen': output[2], 'gencost': output[3], 'branch': output[4] }
# print(output)
# print(mpc['bus'])
# print(mpc['branch'])
"""
for i in range(len(mpc['bus'])):
        mpc['bus'][i][0]+= -1
for i in range(len(mpc['branch'])):
        mpc['branch'][i][0]+= -1
        mpc['branch'][i][1]+= -1
for i in range(len(mpc['gen'])):
        mpc['gen'][i][0] += -1  
        mpc['gencost'][i][0] += -1
#print(mpc['bus'])
#"""
# [Ybus, Yf, Yt] = makeYbus(mpc['baseMVA'],mpc['bus'],mpc['branch'])
# ppopt = ppoption(PF_ALG=2)
# r1 = opf(mpc, ppopt)
# r2 = runpf(mpc, ppopt)

# TODO: creat a clone


# print(r2[0]['bus'])
# print(r1['bus'])
# print(r2[0]['gen'][0][1])
# print(r1['gen'][0][1])
# """
# print(r[0]['bus'])
# print(r[0]['gen'][i][1])

