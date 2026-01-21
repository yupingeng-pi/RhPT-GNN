import logging
import torch.nn as nn
#import torch
from utils import *
#import scipy.io
from training_steps import *
import pandas as pand
from time import time
from datetime import timedelta
from torch.profiler import profile, record_function, ProfilerActivity

PROFILE = False

def is_eval_epoch(cur_epoch, eval_period):
    """Determines if the model should be evaluated at the current epoch."""
    return (
        (cur_epoch + 1) % eval_period == 0
        or cur_epoch == 0
    )


def get_true_costs(loader, characteristics, device): # Solver costs
    """
    This function returns the true costs of all input samples per batch. It is necessary to compute the cost difference between the model and solver. May not be necessary for the future.
    """
    gen_index = torch.tensor(characteristics['gen_index'])
    node_chars = torch.tensor(characteristics['nodes'].values)
    norm_coeffs = characteristics['norm_coeffs']
    total_nb_nodes = characteristics['total_node_nb']
    true_costs = np.zeros(len(loader))
    for i, data in enumerate(loader):
        batch_size = data.num_graphs
        true = data.y
        true_pg = true.T[0]*norm_coeffs['P_norm']
        true_qg = true.T[1]*norm_coeffs['Q_norm']
        true_pg = true_pg.view(batch_size, total_nb_nodes).T
        true_qg = true_qg.view(batch_size, total_nb_nodes).T
        static_costs = torch.full((1,batch_size), characteristics['static_costs'].sum()).squeeze(0)
        true_costs[i] = compute_total_cost(true_pg,true_qg, gen_index, node_chars, batch_size, device, static_costs)
    return true_costs


def train_regr_pinn_correct(loaders,characteristics, model, optimizer, device, eval_period, scheduler, run, training_params, batch_size, augmented_epochs, penalty_multipliers):
    '''
    This is one of the most important functions of all of the project. It defines the training iterations for the augmented Lagrangian method.    
    '''

    # We initialize the previous loss dictionnary, which is needed to update the AL multipliers
    previous_loss_dict = get_zero_constraints(model, loaders[0], characteristics, device, batch_size) #This is very important to start the AL method. Since the AL multipliers update with loss we need initial loss of 0 if we only do AL method


    # Training with the Augmented Lagrangian loss -------------------------------------------------------------------------------------------------------------------------
    
    mun0, muf0, muh0, betaf, betah  = penalty_multipliers #mun0 is for cost balance, the rest are the equality and inequality penalty multipliers
    penalty_multipliers = penalty_multipliers_init(penalty_multipliers)
    
    if run != 0: #Log these parameters into Neptune
        run["parameters/Mun0"] = mun0
        run["parameters/Muf0"] = muf0 
        run["parameters/Muh0"] = muh0
        run["parameters/Betaf"] = betaf
        run["parameters/Betah"] = betah
        run["parameters/AL epochs"] = augmented_epochs

    lambdas = init_lambda(loaders[0],characteristics, device, batch_size) # Initiates a vector full of 0 of the appropriate sizes for the AL multiplers
    true_costs = get_true_costs(loaders[0], characteristics, device)

    t0 = time()
    for epoch in range(augmented_epochs):
        update = False
        if is_eval_epoch(epoch, 100): #We print & send info to Neptune every 100 epochs
            print(" ################################ Epoch {} ################################".format(epoch + 1))
            update = True
        penalty_multipliers, previous_loss_dict = training_step_coo_lagr(loaders[0], characteristics,model, optimizer, previous_loss_dict, lambdas, penalty_multipliers, run, device, true_costs, update)

        if is_eval_epoch(epoch, eval_period):
            eval_epoch_regr(loaders[1], model, device, run, characteristics)
            scheduler.step()

    t1 = time()
    t = str(timedelta(seconds = t1-t0)) 
    print('function inferrence takes: ' + t)

    if run == 0:
        gamma0, lr0, case, epochs = training_params
        hyperparameters = pand.DataFrame(data = [mun0, muf0, muh0, betaf, betah, augmented_epochs, gamma0, lr0, case, epochs], index = ['mun0', 'muf0', 'muh0', 'betan', 'betaf', 'betah', 'epochs', 'gamma0', 'lr0', 'case', 'epochs'])
        pand.DataFrame.to_csv(hyperparameters, 'Results/hyperparameters.csv')

    test_epoch_regr_pinn(loaders[2], model, device, characteristics, run)


def eval_epoch_regr(loader, model, device, run, characteristics): 
    '''
    Function used to evaluate the equality loss of the validation set. Useful to make sure the model can still generalize
    '''
    model.eval()
    norm_coeffs = characteristics['norm_coeffs']
    edge_from_bus = characteristics["edge_to_bus"][0][0]
    edge_to_bus = characteristics["edge_to_bus"][0][1]
    gen_to_bus = characteristics["gen_to_bus"]
    edge_nb = len(characteristics['actual_edges'])
    base_MVA = characteristics['norm_coeffs']['Base_MVA']
    nb_nodes = characteristics['node_nb']
    edge_limits = characteristics['edge_limits']
    G = torch.tensor(characteristics['G']).unsqueeze(1)
    B = torch.tensor(characteristics['B']).unsqueeze(1)
    total_nb_nodes = characteristics['total_node_nb']

    eq_loss_avg = 0
    flow_loss_avg = 0

    for n, data in enumerate(loader):
        data.to(device)
        batch_size = data.num_graphs
        pred = model(data)
        
        pd = data.x[:,0]*norm_coeffs['P_norm']
        qd = data.x[:,1]*norm_coeffs['Q_norm']
    
        pg = pred.T[0]*norm_coeffs['P_norm']
        qg = pred.T[1]*norm_coeffs['Q_norm']
        V = pred.T[2,:]*norm_coeffs['V_norm']
        Theta = pred.T[3,:]*np.pi/180*norm_coeffs['Theta_norm']

        Theta = Theta.view(batch_size, total_nb_nodes).T[:nb_nodes]
        V = V.view(batch_size, total_nb_nodes).T[:nb_nodes]
        pg = pg.view(batch_size, total_nb_nodes).T
        qg = qg.view(batch_size, total_nb_nodes).T

        if characteristics['Ref_node'] >= 0:
            Theta[characteristics['Ref_node'],:] = 0

        pd = pd.view(batch_size, total_nb_nodes).T[:nb_nodes]
        qd = qd.view(batch_size, total_nb_nodes).T[:nb_nodes]

        eq_loss , flow_loss  = compute_eq_loss(pg, qg, pd, qd, V, Theta, edge_from_bus, edge_to_bus, gen_to_bus, nb_nodes, edge_nb, base_MVA, edge_limits, G, B, batch_size, device)

        eq_loss_avg += torch.abs(eq_loss).sum()
        flow_loss_avg += torch.abs(flow_loss).sum()
    
    if run != 0:
        run["val Eq loss"].log(eq_loss_avg/len(loader))
        run["val Flow loss"].log(flow_loss_avg/len(loader))

    return None

#@torch.no_grad()
def test_epoch_regr_pinn(loader, model, device, characteristics, run): #
    '''
    This is the final testing function once the model is trained. We look at equality losses and costs as the main evaluation metrics.
    I tend to print way too much stuff but it's never a bad thing to have more info.
    '''
    norm_coeffs = characteristics['norm_coeffs']
    model.eval()
    eq_losses = []
    true_eq_losses = []
    cost_differences = []
    trues = []
    preds = []
    nb_nodes = characteristics['node_nb']

    edge_from_bus = characteristics["edge_to_bus"][0][0]
    edge_to_bus = characteristics["edge_to_bus"][0][1]
    gen_to_bus = characteristics["gen_to_bus"]
    edge_nb = len(characteristics['actual_edges'])
    base_MVA = characteristics['norm_coeffs']['Base_MVA']
    edge_limits = characteristics['edge_limits']
    gen_index = torch.tensor(characteristics['gen_index'])
    G = torch.tensor(characteristics['G']).unsqueeze(1)
    B = torch.tensor(characteristics['B']).unsqueeze(1)
    node_chars = torch.tensor(characteristics['nodes'].values)
    total_nb_nodes = characteristics['total_node_nb']
    i = 0
    
    for data in loader:
        i += 1
        data.to(device)
        batch_size = data.num_graphs
        true = data.y.detach()#.cpu()
        print('True: ')
        print(true)
        pred = model(data)
        for i in range(characteristics['total_node_nb']*batch_size):
            if i%characteristics['total_node_nb'] not in characteristics['gen_index']:
                pred.T[0][i] = 0
                pred.T[1][i] = 0
            if i%characteristics['total_node_nb'] > characteristics['node_nb']:
                pred.T[2][i] = 0
                pred.T[3][i] = 0

        pd = data.x[:,0]*norm_coeffs['P_norm']
        qd = data.x[:,1]*norm_coeffs['Q_norm']
        pg = pred.T[0]*norm_coeffs['P_norm']
        qg = pred.T[1]*norm_coeffs['Q_norm']
        V = pred.T[2,:]*norm_coeffs['V_norm']
        Theta = pred.T[3,:]*np.pi/180*norm_coeffs['Theta_norm']

        Theta = Theta.view(batch_size, total_nb_nodes).T[:nb_nodes]
        V = V.view(batch_size, total_nb_nodes).T[:nb_nodes]
        pg = pg.view(batch_size, total_nb_nodes).T
        qg = qg.view(batch_size, total_nb_nodes).T
        if characteristics['Ref_node'] >= 0:
            Theta[characteristics['Ref_node'],:] = 0
        static_costs = torch.full((1,batch_size), characteristics['static_costs'].sum()).squeeze(0)

        print('Prediction: ')
        print(pred)

        pd = pd.view(batch_size, total_nb_nodes).T[:nb_nodes]
        qd = qd.view(batch_size, total_nb_nodes).T[:nb_nodes]
        
        eq_loss , flow_loss  = compute_eq_loss(pg, qg, pd, qd, V, Theta, edge_from_bus, edge_to_bus, gen_to_bus, nb_nodes, edge_nb, base_MVA, edge_limits, G, B, batch_size, device) #We probably want to change this to get a better spread
        cost_loss = compute_total_cost(pg,qg, gen_index, node_chars, batch_size, device, static_costs)

        eq_losses.append(abs(eq_loss).detach().sum())

        trues.append(pand.DataFrame(true.detach().numpy()))
        preds.append(pand.DataFrame(pred.detach().numpy()))


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

        static_costs = torch.full((1,batch_size), characteristics['static_costs'].sum()).squeeze(0)
        
        eq_loss_true , flow_loss_true = compute_eq_loss(true_pg, true_qg, pd, qd, true_V, true_Theta, edge_from_bus, edge_to_bus, gen_to_bus, nb_nodes, edge_nb, base_MVA, edge_limits, G, B, batch_size, device)
                
        true_cost = compute_total_cost(true_pg,true_qg, gen_index, node_chars, batch_size, device, static_costs)

        print('Model Eq loss: ' + str(abs(eq_loss).detach().sum()))
        print('True Eq loss: ' + str(abs(eq_loss_true).detach().sum()))
        cost_differences.append(torch.abs(cost_loss.detach() - true_cost.detach()))
        print('True cost: ' + str(true_cost.detach().sum()))
        print('Model cost: ' + str(cost_loss.detach().sum()))

        true_eq_losses.append(abs(eq_loss_true).detach().sum())
    
    avg_true_eq_loss = np.mean(np.array(true_eq_losses))
    avg_eq_loss = np.mean(np.array(eq_losses))
    avg_cost_difference = np.mean(np.array(cost_differences))
    print('Average equality loss: ' + str(avg_eq_loss))
    print('Average cost difference: ' + str(avg_cost_difference))
    print('Transmission limit violations: ' + str(flow_loss))
    print('Average true equality loss: ' + str(avg_true_eq_loss))


    trues = pand.concat(trues)
    preds = pand.concat(preds)

    print(eq_losses) #These are per batch
    print(cost_differences)

    # Save all relevant information and the model to a Results folder
    
    relevant_results_saver = pand.DataFrame(data = np.array([eq_losses, cost_differences]), index = ['eq_losses', 'cost_differences'])
    pand.DataFrame.to_csv(relevant_results_saver, 'Results/relevant_results.csv')

    pand.DataFrame.to_csv(trues, 'Results/true.csv')
    pand.DataFrame.to_csv(preds, 'Results/pred.csv')

    torch.save(model, 'Results/model.pt')

    
