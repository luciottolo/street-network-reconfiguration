#I download the data from server. I simplify and consolidate. I obtain a SCC, and I go to a single layer configuration
#########################################################
import sys; sys.prefix
import os
os.getcwd()
import osmnx as ox
import networkx as nx
import seaborn as sns
import geonetworkx as gnx
import pandas as pd
sns.set(style="darkgrid")


#####################################
#to get the newtork
place = 'Zurich, Switzerland'
G = ox.graph_from_place(place, network_type='drive', simplify=False,  buffer_dist=500, truncate_by_edge=True, clean_periphery=True)

#to save the NW, graphml for py, gpkg for QGIS
ox.save_graphml(G, filepath='./data/Zurich.graphml')
ox.save_graph_geopackage(G, filepath='./data/Zurich.gpkg', encoding='utf-8', directed=True)

#G.order() to get the nodes
#G.size() to get the edges
N, K = G.order(), G.size()
avg_deg = round(float(K) / N , 2)
print('Zurich not simplified')
print ("Nodes: ", N )
print ("Edges: ", K )
print ("Average degree: ", avg_deg)
print ("SCC: ", nx.number_strongly_connected_components(G))
print ("WCC: ", nx.number_weakly_connected_components(G))

#printing on screen or get a latex output for simple tables
ind = ['Nodes','Edges','Average degree', 'SCC', 'WCC']
data = [N, K, avg_deg, nx.number_strongly_connected_components(G),nx.number_weakly_connected_components(G)]

df = pd.DataFrame(data, index = ind)
print(df)

print(df.to_latex(index=True))

# turn off strict mode and see what nodes we'd remove, in yellow
nc = ["r" if ox.simplification._is_endpoint(G, node) else "y" for node in G.nodes()]
fig, ax = ox.plot_graph(G, node_color='b')
fig, ax = ox.plot_graph(G,node_size=0,edge_color='r')
fig, ax = ox.plot_graph(G, node_color=nc)

# simplify the network
G = ox.simplify_graph(G)
fig, ax = ox.plot_graph(G, node_color="r")

N, K = G.order(), G.size()
avg_deg = round(float(K) / N , 2)
print('Zurich simplified from graph')
print ("Nodes: ", N )
print ("Edges: ", K )
print ("Average degree: ", avg_deg)
print ("SCC: ", nx.number_strongly_connected_components(G))
print ("WCC: ", nx.number_weakly_connected_components(G))

ind = ['Nodes','Edges','Average degree', 'SCC', 'WCC']
data = [N, K, avg_deg, nx.number_strongly_connected_components(G),nx.number_weakly_connected_components(G)]

df = pd.DataFrame(data, index = ind)
print(df)

print(df.to_latex(index=True))

G = ox.graph_from_place(place, network_type='drive', simplify=True,  buffer_dist=500, truncate_by_edge=True, clean_periphery=True)

N, K = G.order(), G.size()
avg_deg = round(float(K) / N , 2)
print('Zurich simplified from server')
print ("Nodes: ", N )
print ("Edges: ", K )
print ("Average degree: ", avg_deg)
print ("SCC: ", nx.number_strongly_connected_components(G))
print ("WCC: ", nx.number_weakly_connected_components(G))

ind = ['Nodes','Edges','Average degree', 'SCC', 'WCC']
data = [N, K, avg_deg, nx.number_strongly_connected_components(G),nx.number_weakly_connected_components(G)]

df = pd.DataFrame(data, index = ind)
print(df)

print(df.to_latex(index=True))

ox.save_graphml(G, filepath='./data/zurichsimplified.graphml')
ox.save_graph_geopackage(G, filepath='./data/zurichsimplified.gpkg', encoding='utf-8', directed=True)

#consolidate
G_proj = ox.project_graph(G)

g = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=15, dead_ends=False)

ox.save_graphml(g, filepath='./data/zurichconsolidated.graphml')
ox.save_graph_geopackage(g, filepath='./data/zurichconsolidated.gpkg', encoding='utf-8', directed=True)
fig, ax = ox.plot_graph(g, node_color="r")

N, K = g.order(), g.size()
avg_deg = round(float(K) / N , 2)
print('Zurich consolidated')
print ("Nodes: ", N )
print ("Edges: ", K )
print ("Average degree: ", avg_deg)
print ("SCC: ", nx.number_strongly_connected_components(g))
print ("WCC: ", nx.number_weakly_connected_components(g))

ind = ['Nodes','Edges','Average degree', 'SCC', 'WCC']
data = [N, K, avg_deg, nx.number_strongly_connected_components(g),nx.number_weakly_connected_components(g)]

df = pd.DataFrame(data, index = ind)
print(df)

print(df.to_latex(index=True))

#get SCC

gnx.remove_self_loop_edges(g)
gnx.remove_dead_ends(g)

print(list(nx.strongly_connected_components(g)))
Gcc = max(nx.strongly_connected_components(g), key=len)
giantC = g.subgraph(Gcc)

g = nx.MultiDiGraph(giantC)

N, K = g.order(), g.size()
avg_deg = round(float(K) / N , 2)
print('Zurich strongly connected')
print ("Nodes: ", N )
print ("Edges: ", K )
print ("Average degree: ", avg_deg)
print ("SCC: ", nx.number_strongly_connected_components(g))
print ("WCC: ", nx.number_weakly_connected_components(g))

ind = ['Nodes','Edges','Average degree', 'SCC', 'WCC']
data = [N, K, avg_deg, nx.number_strongly_connected_components(g),nx.number_weakly_connected_components(g)]

df = pd.DataFrame(data, index = ind)
print(df)

print(df.to_latex(index=True))

fig, ax = ox.plot_graph(g, node_color="r")
ox.save_graphml(g, filepath='./data/zurichstronglyconnected.graphml')
ox.save_graph_geopackage(g, filepath='./data/zurichstronglyconnected.gpkg', encoding='utf-8', directed=True)

city = ox.geocode_to_gdf('Zurich, Switzerland')
ax = ox.project_gdf(city).plot()
_ = ax.axis('off')

nodes_proj = ox.graph_to_gdfs(g, edges=False)
graph_area_m = nodes_proj.unary_union.convex_hull.area
print('surface in sqrm:', graph_area_m)

#remove k != 0

# remove k=1
print(len(g.edges))
copygraph = nx.MultiDiGraph(g)

for u, v, key, data in g.edges(data=True, keys=True):
    if key != 0:
        copygraph.remove_edge(u, v, key=key)

g = nx.MultiDiGraph(copygraph)

nx.is_strongly_connected(g)
print(len(g.edges))

N, K = g.order(), g.size()
avg_deg = round(float(K) / N , 2)
print('Zurich 0 layer')
print ("Nodes: ", N )
print ("Edges: ", K )
print ("Average degree: ", avg_deg)
print ("SCC: ", nx.number_strongly_connected_components(g))
print ("WCC: ", nx.number_weakly_connected_components(g))

ind = ['Nodes','Edges','Average degree', 'SCC', 'WCC']
data = [N, K, avg_deg, nx.number_strongly_connected_components(g),nx.number_weakly_connected_components(g)]

df = pd.DataFrame(data, index = ind)
print(df)

print(df.to_latex(index=True))
#some stats
# show some basic stats about the network
df = pd.DataFrame.from_dict(ox.basic_stats(g, area=graph_area_m, clean_int_tol=15))

print(df)

print(df.to_latex(index=True))


#some nice pictures
# convert graph to line graph so edges become nodes and vice versa
edge_centrality = nx.closeness_centrality(nx.line_graph(g))
nx.set_edge_attributes(g, edge_centrality, "edge_centrality")

# color edges in original graph with closeness centralities from line graph
ec = ox.plot.get_edge_colors_by_attr(g, "edge_centrality", cmap="inferno")
fig, ax = ox.plot_graph(g, edge_color=ec, edge_linewidth=2, node_size=0)

#nodes bc
bc = nx.betweenness_centrality(g, weight="length")
max_node, max_bc = max(bc.items(), key=lambda x: x[1])
print(max_node, max_bc)

nc = ["r" if node == max_node else "w" for node in g.nodes]
ns = [80 if node == max_node else 15 for node in g.nodes]
fig, ax = ox.plot_graph(g, node_size=ns, node_color=nc, node_zorder=2)

# add the betweenness centraliy values as new node attributes, then plot
nx.set_node_attributes(g, bc, "bc")
nc = ox.plot.get_node_colors_by_attr(g, "bc", cmap="plasma")
fig, ax = ox.plot_graph(
    g,
    node_color=nc,
    node_size=30,
    node_zorder=2,
    edge_linewidth=0.2,
    edge_color="w",
)


#the same but with edge betweenness centrality
edge_bbcentrality = nx.edge_betweenness_centrality(g, weight='lenght')
nx.set_edge_attributes(g, edge_bbcentrality, "edge_bbcentrality")

# color edges in original graph with bb centralities from graph
ec = ox.plot.get_edge_colors_by_attr(g, "edge_bbcentrality", cmap="inferno")
fig, ax = ox.plot_graph(g, edge_color=ec, edge_linewidth=2, node_size=0)

ox.save_graphml(g, filepath='./data/zurichstronglyconnectedwbc.graphml')
ox.save_graph_geopackage(g, filepath='./data/zurichstronglyconnectedwbc.gpkg', encoding='utf-8', directed=True)


##################################################################
#descriptive stat in next doc
