import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

"""
Plots the data for M = 1 and inference states.
"""


# Paths
paths = ['case9-OP-4941', 'case24-OP-5277', 'case30-OP-5315', 'case118-OP-5278']
differences = []
for case in paths:
    difference = (pd.read_csv("Results\paper_results\M=1\{}\pred.csv".format(case)) - pd.read_csv("Results\\paper_results\\M=1\\{}\\true.csv".format(case))).abs()
    difference = difference.drop(difference.columns[0], axis = 1)
    differences.append(difference)

norms = []
gen_indices = []
total_nb_nodes = []

for case in ['case9','case24_ieee_rts','case30','case118']:
    with open('Input_Data/characteristics_{}.pkl'.format(case), 'rb') as f:
        characteristics = pickle.load(f)
        norms.append(characteristics['norm_coeffs'])
        gen_indices.append(characteristics['gen_index'])
        total_nb_nodes.append(characteristics['total_node_nb'])


for i in range(len(paths)):
    differences[i] = differences[i].mul(list(norms[i].values())[:4], axis=1).mean(axis=0)
    differences[i][0] *= total_nb_nodes[i]/len(gen_indices[i])
    differences[i][1] *= total_nb_nodes[i]/len(gen_indices[i]) #Here we make sure that we only average over the buses on which there is a generator by applying a nb_nodes/nb_gens ratio

differences[3][3] = 0

data = pd.DataFrame(differences).values.tolist()

quantities = ['Pg [MW]', 'Qg [MVAr]', 'V [p.u.]', 'Theta [deg]']
cases = ['IEEE9','IEEE24','IEEE30','IEEE118']
# Define color palette
pal_node = sns.color_palette("Dark2", 4)  # 4 colors for 4 quantities

# Plot for node
plt.figure(figsize=(10, 6))
for i, quantity in enumerate(quantities):
    plt.bar(np.arange(4) + i * 0.2, data[i], width=0.2, color=pal_node[i], label=cases[i])
plt.xlabel('Quantity', fontsize=16)
plt.ylabel('Average absolute difference', fontsize=16)
plt.xticks(np.arange(4) + 0.3, ['Pg', 'Qg', 'V', 'T'], fontsize=16)
plt.yticks(fontsize=16)
plt.yscale('log')
plt.legend(fontsize=16)
plt.show()


"""
# For inferrence times -----------------------------------------------------------------------------------------------
# Here you have to hard-include the data    
solver_times = [0.59, 0.60, 0.61, 0.70]
model_times = [0.0034, 0.0021, 0.0031, 0.0031]

data = [solver_times, model_times]
data = list(map(list, zip(*data)))

quantities = ['Solver time', 'Model time']
cases = ['IEEE9', 'IEEE24', 'IEEE30', 'IEEE118']

df = pd.DataFrame(data, index=cases, columns=quantities)
print(df)

# Define color palette
pal_node = sns.color_palette("Dark2", 2)  # 2 colors for 2 quantities

# Plot for node
plt.figure(figsize=(10, 6))
bar_width = 0.35

for i, quantity in enumerate(quantities):
    plt.bar(np.arange(len(cases)) + i * bar_width, df[quantity], width=bar_width, color=pal_node[i], label=quantity)

plt.xlabel('Cases', fontsize=16)
plt.ylabel('Time (s)', fontsize=16)
plt.xticks(np.arange(len(cases)) + bar_width / 2, cases, fontsize=16)
plt.yticks(fontsize=16)
plt.yscale('log')
plt.legend(fontsize=16)
plt.show()

#"""
