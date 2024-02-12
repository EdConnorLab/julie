import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from monkeyids import Monkey


G = nx.DiGraph()
monkeys = [Monkey.Z_M1, Monkey.Z_F1, Monkey.Z_F2, Monkey.Z_F3, Monkey.Z_F4, Monkey.Z_F5, Monkey.Z_F6, Monkey.Z_F7, Monkey.Z_J1, Monkey.Z_J2]
labels = {monkey: monkey.value for monkey in monkeys}
nodes = range(len(monkeys))
G.add_nodes_from(nodes)
nx.set_node_attributes(G, labels, 'id')
pos = nx.spring_layout(G)
node_size = 1000
print(G.nodes())
print(G.nodes(data=True))



min_value = 0
max_value = 3
# matrix = np.random.randint(min_value, max_value+1, size=(10, 10))
matrix = np.random.rand(10,10)
np.fill_diagonal(matrix, 0)
# Add directed edges with weights based on the matrix
for i in nodes:
    for j in nodes:
        weight = matrix[i][j]
        if weight != 0:
            G.add_edge(i, j, weight=weight)

# Create a layout for the graph (e.g., circular layout)
pos = nx.circular_layout(G)

# Extract edge weights
edge_weights = [G[i][j]["weight"] for i, j in G.edges()]

# Draw nodes and edges with custom edge widths based on weights
nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue")
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color="gray", connectionstyle="arc3,rad=0.1", arrowsize=15)

plt.show()
# # Add edges with weights from the matrix
# for i in range(len(nodes)):
#     for j in range(len(nodes)):
#         weight = matrix[i][j]
#         print(weight)
#         # if weight != 0 and i != j:  # Skip self-loops and edges with weight 0
#         #     G.add_edge(i, j, weight=weight)
#         #     G.add_edge(j, i, weight=weight)  # Bidirectional edge
#
# print(matrix)
# edge_weights = nx.get_edge_attributes(G, 'weight')
# print(edge_weights)
# nx.draw(G, labels=labels, with_labels=True, node_size=node_size)
# plt.show()
# print(G.nodes())
# print(G.nodes(data=True))