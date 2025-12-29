import pandapower as pp
import pandapower.networks as pn
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import *
import pickle

device = "cpu"

"""
This file is a bit of a mess but it works in the following way:
    1. Loads a trained model and test set
    2. Computes the different losses and flows using functions from the utils.py file
    3. Plots them in a power flow graph
"""


# Main User Defined Inputs --------------------------------------------------------------------------------------------------
case = "case9"
points = 1
batch_size = 1

path = "Results\paper_results\M=2+\case9_M=500\model.pt"
#test_loader = torch.load("Input_Data/test_loader_{}_{}_{}.pt".format(case, points, batch_size))
test_loader = torch.load("Input_Data/test_loader_{}_test_{}_{}.pt".format(case, points, batch_size))
#test_loader = torch.load("Input_Data/test_loader_{}_test_N-1_{}_{}.pt".format(case, points, batch_size))

# ---------------------------------------------------------------------------------------------------------------------------


with open('Input_Data/characteristics_{}.pkl'.format(case), 'rb') as f:
#with open('Input_Data/characteristics_{}_test_N-1.pkl'.format(case), 'rb') as f:
            characteristics = pickle.load(f)

model = torch.load(path)
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

print(characteristics['actual_edges'])

for data in test_loader:
    norm_coeffs = characteristics['norm_coeffs']
    true = data.y
    pred = model(data)
    
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

    Theta = Theta.view(batch_size, nb_nodes).T[:nb_nodes]
    V = V.view(batch_size, nb_nodes).T[:nb_nodes]
    pg = pg.view(batch_size, nb_nodes).T
    qg = qg.view(batch_size, nb_nodes).T
    Theta[characteristics['Ref_node'],:] = 0

    pd = pd.view(batch_size, nb_nodes).T[:nb_nodes]
    qd = qd.view(batch_size, nb_nodes).T[:nb_nodes]



    print('True: ')
    print(true)

    print()

    print('Pred: ')
    print(pred)



    #eq_loss , flow_loss  = compute_eq_loss(pg, qg, pd, qd, V, Theta, edge_from_bus, edge_to_bus, gen_to_bus, nb_nodes, edge_nb, base_MVA, edge_limits, G, B, batch_size, device) #We probably want to change this to get a better spread
    #plate_loss = torch.abs(torch.sum(pd) - pg[0] - pg[1] - pg[2])
    #cost_loss = compute_total_cost(pg,qg, gen_index, node_chars, batch_size, device)
    p_f, q_f = compute_flows(V, Theta, edge_from_bus, edge_to_bus, base_MVA, edge_limits, G, B, batch_size, device)

    #print(eq_loss)
    #print(p_f)

#"""

#"""


import pandapower as pp
import pandapower.networks as pn
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import matplotlib as mpl
from svgpathtools import svg2paths
from svgpath2mpl import parse_path

from convert_Base64 import return_URI

def generate_marker_from_svg(svg_path):
    image_path, attributes = svg2paths(svg_path)

    image_marker = parse_path(attributes[0]['d'])

    image_marker.vertices -= image_marker.vertices.mean(axis=0)

    image_marker = image_marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
    image_marker = image_marker.transformed(mpl.transforms.Affine2D().scale(-1,1))

    return image_marker

# Create the network
net = pn.case9()
gen_marker = generate_marker_from_svg("gen_symbol.svg")

# Run power flow to calculate voltages and line flows
#pp.runpp(net)

# Get the bus voltages
#voltages = net.res_bus.vm_pu.values
voltages = V.detach().numpy().reshape(-1)
#print(voltages)
#print(voltages)

# Normalize the voltages to create a colormap
voltage_norm = plt.Normalize(vmin=min(voltages), vmax=max(voltages))
voltage_colormap = plt.cm.plasma  # You can choose any colormap you prefer

# Map the normalized voltages to colors
voltage_colors = [voltage_colormap(voltage_norm(voltage)) for voltage in voltages]
voltage_plotly_colors = ['rgb({}, {}, {})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in voltage_colors]

# Get the line flows
#line_flows = abs(net.res_line.p_from_mw.values)  # absolute values to make all flows positive
#print(line_flows)
line_flows = abs(p_f.detach().numpy().reshape(-1)[::2])

# Normalize the line flows to create a colormap
flow_norm = plt.Normalize(vmin=min(abs(line_flows)), vmax=max(abs(line_flows)))
flow_colormap = plt.cm.turbo  # You can choose any colormap you prefer

# Map the normalized line flows to colors
flow_colors = [flow_colormap(flow_norm(flow)) for flow in abs(line_flows)]
flow_plotly_colors = ['rgb({}, {}, {})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in flow_colors]

# Get the bus geodata
bus_geodata = net.bus_geodata
#print(bus_geodata) #Keep this

# Create a plotly figure
fig = go.Figure()

# Add lines to the plot
for idx, line in net.line.iterrows():
    from_bus = line.from_bus
    to_bus = line.to_bus
    x_coords = [bus_geodata.x.at[from_bus], bus_geodata.x.at[to_bus]]
    y_coords = [bus_geodata.y.at[from_bus], bus_geodata.y.at[to_bus]]

    flow = line_flows[idx]
    color = flow_plotly_colors[idx]

    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='lines',
        line=dict(width=2, color=color),
        hoverinfo='none'  # Hide hoverinfo for line
    ))

gen_index = [0,1,2]
load_index = [4,6,8]
#svg_data_uri = return_URI()
URIs = return_URI()

# Add buses to the plot with higher z-order
for idx, bus in net.bus.iterrows():
    shape = 'circle'
    #elif bus.name in load_index:
    #    shape = 'triangle-up'
    x_loc =[bus_geodata.x.at[idx]]
    y_loc =[bus_geodata.y.at[idx]]

    fig.add_trace(go.Scatter(
        x=x_loc,
        y=y_loc,
        mode='markers+text',
        marker = dict(symbol = shape, size=20, color=voltage_plotly_colors[idx]),
        text=str(idx),
        textposition='top center',
        hoverinfo='text'
    ))
    if bus.name in gen_index:
        svg_data_uri = URIs['gen']
        fig.add_layout_image(
            dict(
                source=svg_data_uri,
                xref="x",
                yref="y",
                x= x_loc[0] - 0.099,
                y= y_loc[0] - 0.02,
                sizex=0.5,
                sizey=0.5,
                layer="above")
        )
    elif bus.name in load_index:
        svg_data_uri = URIs['load']
        fig.add_layout_image(
            dict(
                source=svg_data_uri,
                xref="x",
                yref="y",
                x= x_loc[0] - 0.099,
                y= y_loc[0] + 0.032,
                sizex=0.5,
                sizey=0.5,
                layer="above")
        )
        

# Add colorbars to the plot
voltage_colorbar_trace = go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    marker=dict(
        colorscale='Plasma',
        cmin=min(voltages),
        cmax=max(voltages),
        colorbar=dict(
            title="Voltage (p.u.)",
            titleside="right",
            x=1.1  # Position the colorbar to the right
        ),
        size=0
    ),
    hoverinfo='none'
)
fig.add_trace(voltage_colorbar_trace)

flow_colorbar_trace = go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    marker=dict(
        colorscale='Turbo',
        cmin=min(line_flows),
        cmax=max(line_flows),
        colorbar=dict(
            title="Line Flow (MW)",
            titleside="right",
            x=1.2  # Position the colorbar to the right of the voltage colorbar
        ),
        size=0
    ),
    hoverinfo='none'
)
fig.add_trace(flow_colorbar_trace)

# Update layout
fig.update_layout(
    title='Power System Network with Generators and Loads',
    plot_bgcolor='white',  # Set background color to white
    xaxis=dict(visible=False),  # Hide x-axis
    yaxis=dict(visible=False),  # Hide y-axis
    margin=dict(l=20, r=200, t=50, b=20),  # Adjust margins to make space for colorbars
    showlegend = False
)

# Show the plot
fig.show()

def demo_generation(svg_path):
    gen_marker = generate_marker_from_svg(svg_path)

    x = np.linspace(0,2*np.pi,10)
    y = np.sin(x)

    plt.plot(x,y,'o',marker=gen_marker,markersize=30)

    plt.show()

#demo_generation("gen_symbol.svg")

