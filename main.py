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

#25/04 clean the df
# Load Edges into GeoDataFrame (to be tried w G2 and GGG)
nodes, edges = ox.graph_to_gdfs(G2)
edges.head()

#present in the lanes three types of objects: lists (when lanes change in the same street), float (NaN), str (number of lanes)
#if list we took the first value (assumption)
#if float (NaN) we substituted with 1
#if str it becames float

for lines in range(len(edges)):
    if type(list(edges['lanes'])[lines])==list:
        edges['lanes'].iloc[lines]=edges['lanes'].iloc[lines][0]
    elif type(list(edges['lanes'])[lines])==float:
        edges['lanes'].iloc[lines]=1
    elif type(list(edges['lanes'])[lines])==str:
        edges['lanes'].iloc[lines]=float(edges['lanes'].iloc[lines])


#in width also present different objects kind. If list take first value, 
#if float (NaN) use lanes*3.25m (Swissstandard), otherwise make number

for lines in range(len(edges)):
    if type(list(edges['width'])[lines])==list:
        edges['width'].iloc[lines]=edges['width'].iloc[lines][0]
    elif type(list(edges['width'])[lines])==float:
        edges['width'].iloc[lines]=edges['lanes']*3.25
    elif type(list(edges['width'])[lines])==str:
        edges['width'].iloc[lines]=float(edges['width'].iloc[lines])


#create new columns, Area, new width (shrink to one lane what is not one lane), new Area  

edges['area'] = edges['length'] * edges['width']

edges['width new']=edges['width']/edges['lanes']

edges['area new']= edges['length'] * edges['width new']

sum(list(edges['area new']))

#calculate area shrink %

(float(sum(list(edges['area']))[0][2])-float(sum(list(edges['area new']))[0][2]))/float(sum(list(edges['area']))[0][2])

# Modify oneway from Boolean to int
edges["oneway"] = edges["oneway"].astype(int)

#df is now complete, with both not shrinked and shrinked roads. Now QGIS visualization and Algorithms.


