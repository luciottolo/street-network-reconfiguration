#Cleancode, other half done things in docs w date

import networkx as nx
import osmnx as ox
import json
import time
from itertools import chain

G = ox.graph_from_place('Zurich, Switzerland', network_type='drive')
fig, ax = ox.plot_graph(G, node_color="r")

#G = ox.load_graphml("./data/network.graphml")
#ox.plot_graph(G)

print("node count:", len(G.nodes()))
print("edge count:", len(G.edges()))

# add travel time based on maximum speed
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

print(set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True))))

print(list(list(G.edges(data=True))[0][-1].keys()))

G_proj = ox.project_graph(G)
G2 = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=15, dead_ends=False)#or true
print(len(G2))
fig, ax = ox.plot_graph(G2, node_color="r")

time_start = time.process_time()

print(sum(nx.betweenness_centrality(G2, endpoints=True)))
#run your code
time_elapsed = (time.process_time() - time_start)
print(time_elapsed)

# show the simplified network with edges colored by length
ec = ox.plot.get_edge_colors_by_attr(G2, attr="length", cmap="plasma_r")
fig, ax = ox.plot_graph(G2, node_color="w", node_edgecolor="k", node_size=50, edge_color=ec, edge_linewidth=3)

# highlight all parallel (multiple) edges
ec = ["gray" if k == 0 or u == v else "r" for u, v, k in G2.edges(keys=True)]
fig, ax = ox.plot_graph(G2, node_color="w", node_edgecolor="k", node_size=50, edge_color=ec, edge_linewidth=3
)

# highlight all one-way edges in the mission district network from earlier
ec = ["r" if data["oneway"] else "w" for u, v, key, data in G2.edges(keys=True, data=True)]
fig, ax = ox.plot_graph(G2, node_size=0, edge_color=ec, edge_linewidth=1.5, edge_alpha=0.7)

GGG = nx.convert_node_labels_to_integers(G2, first_label=1)
print(GGG.nodes(data=True))
print(GGG.edges(data=True))

print(GGG)

#investigate a shipy.sparse what is
print(nx.adjacency_matrix(GGG))

# Drop Unwanted Columns to clean the tags
#edges.drop(['oneway', 'lanes', 'maxspeed'], inplace=True, axis=1)


