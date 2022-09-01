#here is as ThesisZurich4a but bc has been weighted by edge length
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import seaborn as sns
import geonetworkx as gnx
import netgraph
sns.set(style="darkgrid")
from operator import itemgetter
import time

#decide the graph to upload here
filepath = './data/zurichfilledtags.graphml'
g = ox.load_graphml(filepath)

nodes, edges = ox.graph_to_gdfs(g)

#i check the tags are still of the same type of be4
df = edges['lanes'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.lanes.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

df = edges['width'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.width.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#from here I understand osmnx does not preserve tags class, gonnachangethemmanually

for u,v,data in g.edges(data=True):
    data['width']=float(data['width'])
    data['lanes']=int(data['lanes'])

#calculate area of SN
#since when u=v and v=u information repeat, I will create a sub class to calculate area considering
#only when they don't.

graphtocalcarea = nx.MultiDiGraph(g)

set_list = [set(a) for a in graphtocalcarea.edges()] # collect all edges, lose positional information
remove_list = [] # initialise

for i in range(len(set_list)):
    edge = set_list.pop(0) # look at zeroth element in list:

    # if there is still an edge like the current one in the list,
    # add the current edge to the remove list:
    if set_list.count(edge) > 0:
        u,v = edge

        # add the reversed edge
        remove_list.append((v, u))

        # alternatively, add the original edge:
        # remove_list.append((u, v))

graphtocalcarea.remove_edges_from(remove_list) # remove all edges collected above

#add area variable sqrm
area = {(u,v,key) : data['length']*data['width'] for u, v, key, data in graphtocalcarea.edges(keys=True, data=True)}

nx.set_edge_attributes(graphtocalcarea, area, 'area')

#print(graphtocalcarea.edges(data='area'))
#calculate total street NW area

nodes, edges = ox.graph_to_gdfs(graphtocalcarea)

totwidth = edges['width'].sum()
totarea = edges['area'].sum()

print(totwidth)
print(totarea)

#graph of onewaystreets
ec = ["r" if data["oneway"] == True else "w" for u, v, key, data in g.edges(keys=True, data=True)]
fig, ax = ox.plot_graph(g, node_size=0, edge_color=ec, edge_linewidth=1.5, edge_alpha=0.7)

# to check that oneway tag makes sense
'''
for u, v, data in g.edges(data=True):
    if not (g.has_edge(u, v) and g.has_edge(v, u)):
        print(data['oneway'])

for u, v, data in g.edges(data=True):
    if g.has_edge(u, v) and g.has_edge(v, u):
        print(data['oneway'])
'''

nodes, edges = ox.graph_to_gdfs(g)
df = edges['oneway'].value_counts()
print(df)
print(df.to_latex(index=True))

for u,v,data in g.edges(data=True):
    if data['oneway']==False:
        print(data['lanes'])



#add edges bc as tag

bb = nx.edge_betweenness_centrality(g, normalized=True, weight='length')
nx.set_edge_attributes(g, bb, "betweenness")

######################
#here i shrink
removed_edges = []

gcopy = nx.MultiDiGraph(g)
gcopycopy = nx.MultiDiGraph(g)

#remove two ways streets based on bc, storing eliminated ones in a list of tuples

for u,v,data in sorted(g.edges(data=True), key=lambda t: t[2].get('betweenness')):
    try:
        if g.has_edge(u,v) and g.has_edge(v,u):
            gcopycopy.remove_edge(u,v)
            if nx.is_strongly_connected(gcopycopy):
                gcopy = nx.MultiDiGraph(gcopycopy)
                removed_edges.append((u,v)) #impo here!! added 05/06
            else:
                g.copycopy= nx.MultiDiGraph(gcopy)
    except KeyError:
        pass

g = nx.MultiDiGraph(gcopy)


reverse_edges = []

for x in removed_edges:
    reverse_edges.append(x[::-1])

nodes, edges = ox.graph_to_gdfs(g)
df = edges['oneway'].value_counts()
print(df)
print(df.to_latex(index=True))

# dieting tags
#
# I create the new tags
newlanes = []
nx.set_edge_attributes(g, newlanes, "newlanes")

newwidth = []
nx.set_edge_attributes(g, newwidth, "newwidth")

# I adjust the oneway tags after dieting
# here lood arrow in notes

for u, v, data in g.edges(data=True):
    if (u, v) in reverse_edges:
        data['oneway'] = True

for u, v, data in g.edges(data=True):
    if data['oneway'] == False:
        data['newlanes'] = 2
    else:
        data['newlanes'] = 1

for u, v, data in g.edges(data=True):
    if data['newlanes'] == 1:
        data['newwidth'] = 3.25
    if data['newlanes'] == 2:
        data['newwidth'] = 6.5

nodes, edges = ox.graph_to_gdfs(g)

df = edges['newlanes'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.newlanes.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

df = edges['newwidth'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges['width'].value_counts()
print(df)
print(df.to_latex(index=True))

#calculate area of SN
#since when u=v and v=u information repeat, I will create a sub class to calculate area considering
#only when they don't.


graphtocalcarea = nx.MultiDiGraph(g)

set_list = [set(a) for a in graphtocalcarea.edges()] # collect all edges, lose positional information
remove_list = [] # initialise

for i in range(len(set_list)):
    edge = set_list.pop(0) # look at zeroth element in list:

    # if there is still an edge like the current one in the list,
    # add the current edge to the remove list:
    if set_list.count(edge) > 0:
        u,v = edge

        # add the reversed edge
        remove_list.append((v, u))

        # alternatively, add the original edge:
        # remove_list.append((u, v))

graphtocalcarea.remove_edges_from(remove_list) # remove all edges collected above

#add area variable sqrm
newarea = {(u,v,key) : data['length']*data['newwidth'] for u, v, key, data in graphtocalcarea.edges(keys=True, data=True)}

nx.set_edge_attributes(graphtocalcarea, newarea, 'newarea')

print(graphtocalcarea.edges(data='newarea'))
#calculate total street NW area

nodes, edges = ox.graph_to_gdfs(graphtocalcarea)

totnewwidth = edges['newwidth'].sum()
totnewarea = edges['newarea'].sum()

print(totwidth)
print(totnewarea)

print('width:',totwidth,'  newwidth:', totnewwidth)
print('area:',totarea,'  newarea', totnewarea)

print('diffareas', totarea-totnewarea)

percentage = (totarea-totnewarea)/totarea

ind = ['Width','New width','Area', 'New Area ', 'Percentage Area']
data = [totwidth, totnewwidth, totarea, totnewarea, percentage]

df = pd.DataFrame(data, index = ind)
print(df)
print(df.to_latex(index=True))

for u,v,data in g.edges(data=True):
    if (g.has_edge(u,v) and g.has_edge(v,u)):
        data['oneway']=False
    else:
       data['oneway']=True

#othe oneway graph et similia
ec = ["r" if data["oneway"] == True else "w" for u, v, key, data in g.edges(keys=True, data=True)]
fig, ax = ox.plot_graph(g, node_size=0, edge_color=ec, edge_linewidth=1.5, edge_alpha=0.7)

ec = ox.plot.get_edge_colors_by_attr(g, attr="betweenness", cmap="plasma_r")
fig, ax = ox.plot_graph(g, node_color="w", node_edgecolor="k", node_size=0.1, edge_color=ec, edge_linewidth=3)

ox.save_graphml(g, filepath="./data/zurichdietedlengthasweigth.graphml")
ox.save_graph_geopackage(g, filepath='./data/zurichdietedlengthasweight.gpkg', encoding='utf-8', directed=True)


