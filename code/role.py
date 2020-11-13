from graphrole import RecursiveFeatureExtractor, RoleExtractor
from pprint import pprint
import h5py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import pickle
import numpy as np

import networkx as nx

if __name__ == '__main__':
    # with open("./adj_mx_bay.pkl", 'rb') as f:
    with open("./adj_mx_la.pkl", 'rb') as f:    
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        adj_bay = u.load()
    adj_bay = np.array(adj_bay)[-1]
    G = nx.Graph(adj_bay)

    feature_extractor = RecursiveFeatureExtractor(G)
    features = feature_extractor.extract_features()
    print('features\n', features)

    role_extractor = RoleExtractor(n_roles=None)
    role_extractor.extract_role_factors(features)

    node_roles = role_extractor.roles
    print('\nNode role assignments:')
    pprint(node_roles)

    print('\nNode role membership by percentage:')
    print(role_extractor.role_percentage.round(2))

    # build color palette for plotting
    unique_roles = sorted(set(node_roles.values()))
    color_map = sns.color_palette('Paired', n_colors=len(unique_roles))
    # map roles to colors
    role_colors = {role: color_map[i] for i, role in enumerate(unique_roles)}
    # build list of colors for all nodes in G
    node_colors = [role_colors[node_roles[node]] for node in G.nodes]

    # plot graph
    plt.figure()
    with warnings.catch_warnings():
        # catch matplotlib deprecation warning
        warnings.simplefilter('ignore')
        nx.draw(
            G,
            pos=nx.spring_layout(G, seed=42),
            # with_labels=True,
            node_color=node_colors,
            node_size= 100, #default=300
            font_size = 5 #default=12
        )
    plt.show()