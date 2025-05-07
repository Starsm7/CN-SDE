import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.colors import ListedColormap

custom_colors = ['#8dd3c7', '#f0d4c1', '#bebada', '#ffffb3', '#fccde5',
    '#b3de69', '#fdb462', '#d9d9d9', '#fb8072', '#bc80bd',
    '#ccebc5', '#80b1d3']
custom_cmap = ListedColormap(custom_colors)
# create_cell_fan_heatmap(adj_matrix, cmap='RdBu')
# create_cell_fan_heatmap(adj_matrix, cmap='RdYlBu')

def create_cell_fan_heatmap(matrix, index=0, cmap=custom_cmap):
    if index == 7:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xticks(np.arange(0.5, matrix.shape[1], 1))  # Ticks at centers
    ax.set_yticks(np.arange(0.5, matrix.shape[0], 1))
    ax.set_xticklabels(np.arange(1, matrix.shape[1] + 1, 1))
    ax.set_yticklabels(np.arange(1, matrix.shape[0] + 1, 1))
    ax.tick_params(axis='both', which='both', length=0)

    # Remove spines
    for edge, spine in ax.spines.items():
        spine.set_color('gray')
        # spine.set_visible(False)

    # Add grid at integer positions (cell boundaries)
    ax.set_xticks(np.arange(0, matrix.shape[1] + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, matrix.shape[0] + 1, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)  # Changed to black and thicker lines
    ax.set_axisbelow(True)

    # Set limits to ensure all cells are visible
    ax.set_xlim(0, matrix.shape[1])
    ax.set_ylim(0, matrix.shape[0])

    cmap = plt.get_cmap(cmap)

    for (i, j), val in np.ndenumerate(matrix):
        angle = (1 - val) * 360
        x, y = j + 0.5, i + 0.5  # Center of the cell
        radius = 0.4

        # Draw the circular outline first (will be behind the wedge)
        circle = Circle(
            (x, y),
            radius,
            facecolor='none',
            edgecolor=cmap(val),
            linewidth=1)
        ax.add_patch(circle)

        if angle > 0:
            wedge = Wedge(
                (x, y),
                radius,
                90,  # Start angle (12 o'clock position)
                90 - angle,  # End angle (counter-clockwise)
                facecolor=cmap(val),
                edgecolor='black',  # No edge for the wedge itself
                linewidth=1)
            ax.add_patch(wedge)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    if index == 7:
        plt.colorbar(sm, ax=ax, label='Value', shrink=0.8)

    plt.title(f'Time {index}', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'PCA/Heatmap{index}.png', dpi=1000)

index_list = [1, 2, 3, 4, 5, 6, 7]

for index in index_list:

    network = torch.load(f'leaveout{index}/sample_graph.pt')
    adj_matrix = network.cpu().detach().numpy()

    G = nx.from_numpy_array(adj_matrix)
    edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') if w < 1 or u == v]
    G.remove_edges_from(edges_to_remove)
    plt.figure(figsize=(12, 10))
    # pos = nx.circular_layout(G, scale=0.8)
    pos = nx.spring_layout(G, k=0.2, iterations=50)

    node_degree = dict(G.degree())
    node_size = [v * 100 for v in node_degree.values()]
    node_color = [v for v in node_degree.values()]
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color,
                           cmap=plt.cm.plasma, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', font_weight='bold')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
    # plt.cm.viridis plt.cm.plasma plt.cm.cividis plt.cm.Blues
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Node Degree', fraction=0.025, aspect=20)
    cbar.set_label('Node Degree', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.axis('off')
    plt.savefig(f'PCA/Network_Graph{index}.png', dpi=1000, bbox_inches='tight')

    create_cell_fan_heatmap(adj_matrix, index=index)
