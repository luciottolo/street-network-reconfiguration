import sys; sys.prefix
import os
os.getcwd()
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
from numpy import arange
from matplotlib import pyplot
import random
import numpy as np
from itertools import chain

#########################################
filepath = './data/zurichstronglyconnectedwbc.graphml'
G = ox.load_graphml(filepath)

#here I check what's in the columns
print(set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True))))
print(list(list(G.edges(data=True))[0][-1].keys()))

attrtobekept = ['highway', 'oneway', 'lanes', 'length', 'width', 'osmid', 'name', 'edge_bbcentrality', 'geometry']

#here I go pandas, gettin only columns of interest
nodes, edges = ox.graph_to_gdfs(G)

edges = edges[edges.columns.intersection(attrtobekept)]

g = ox.graph_from_gdfs(nodes, edges)

#after the simplification I notice that lists are created when edges got joint
#I assume neglectible to have two different values, therefore I select
# par conditio I select the first value
#also, I have to have the values types respecting what they mean

#highway

nodes, edges = ox.graph_to_gdfs(g)

null_df = edges.apply(lambda x: sum(x.isnull())).to_frame(name='count')
df = null_df
print(df)
print(df.to_latex(index=True))

df = edges['highway'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.highway.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#clean the highway tag
for u,v,key,data in g.edges(data=True,keys=True):
    try:
        if type(data['highway'])==list:
            data['highway']=data['highway'][0]
        elif type(data['highway'])==str:
            pass
        else:
            data[('highway')]=='otherway'
    except KeyError:
        data['highway']='otherway'

nodes, edges = ox.graph_to_gdfs(g)

null_df = edges.apply(lambda x: sum(x.isnull())).to_frame(name='count')
df = null_df
print(df)
print(df.to_latex(index=True))

df = edges['highway'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.highway.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#lanes

nodes, edges = ox.graph_to_gdfs(g)

null_df = edges.apply(lambda x: sum(x.isnull())).to_frame(name='count')
df = null_df
print(df)
print(df.to_latex(index=True))

df = edges['lanes'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.lanes.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))


# clean the lanes tag.
for u, v, key, data in g.edges(data=True, keys=True):
    if data['highway'] == 'primary':
        try:
            if type(data['lanes']) == list:
                data['lanes'] = data['lanes'][0]
            elif type(data['lanes']) == float:
                data['lanes'] = 2
            elif type(data['lanes']) == int:
                pass
        except KeyError:
            data['lanes'] = 2

    else:
        try:
            if type(data['lanes']) == list:
                data['lanes'] = data['lanes'][0]
            elif type(data['lanes']) == float:
                if g.has_edge(u, v) and g.has_edge(v, u):
                    data['lanes'] = 2
                else:
                    data['lanes'] = 1
            elif type(data['lanes']) == int:
                pass
        except KeyError:
            if g.has_edge(u, v) and g.has_edge(v, u):
                data['lanes'] = 2
            else:
                data['lanes'] = 1

for u, v, key, data in g.edges(data=True, keys=True):
    if type(data['lanes']) == str:
        data['lanes'] = int(data['lanes'])
    else:
        pass

nodes, edges = ox.graph_to_gdfs(g)

null_df = edges.apply(lambda x: sum(x.isnull())).to_frame(name='count')
print(null_df)

df = edges['lanes'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.lanes.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#width

nodes, edges = ox.graph_to_gdfs(g)

null_df = edges.apply(lambda x: sum(x.isnull())).to_frame(name='count')
df = null_df
print(df)
print(df.to_latex(index=True))

df = edges['width'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.width.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

# clean the width

nodes, edges = ox.graph_to_gdfs(g)

edges.width = edges.width.fillna('filled')

for u, v, key, data in g.edges(data=True, keys=True):
    try:
        if type(data['width']) == list:
            data['width'] = data['width'][0]
        elif type(data['width']) == float:
            pass
        elif type(data['width']) == int:
            pass
        else:
            data['width'] = data['lanes'] * 3.25
    except KeyError:
        data['width'] = data['lanes'] * 3.25

for u, v, data in g.edges(data=True):
    if type(data['width']) == str:
        data['width'] = float(data['width'])
    else:
        pass

nodes, edges = ox.graph_to_gdfs(g)

null_df = edges.apply(lambda x: sum(x.isnull())).to_frame(name='count')
df = null_df
print(df)
print(df.to_latex(index=True))

df = edges['width'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.width.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))


# clean the osmid

nodes, edges = ox.graph_to_gdfs(g)

edges.osmid = edges.osmid.fillna('filled')

for u, v, key, data in g.edges(data=True, keys=True):
    try:
        if type(data['osmid']) == list:
            data['osmid'] = data['osmid'][0]
        elif type(data['osmid']) == float:
            pass
        elif type(data['osmid']) == int:
            pass
        else:
            data['osmid'] = 0
    except KeyError:
        data['osmid'] = 0


nodes, edges = ox.graph_to_gdfs(g)

null_df = edges.apply(lambda x: sum(x.isnull())).to_frame(name='count')
df = null_df
print(df)
print(df.to_latex(index=True))

df = edges['osmid'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.osmid.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))



ox.save_graphml(g, filepath = './data/zurichfilledtags.graphml')
ox.save_graph_geopackage(g, filepath='./data/zurichfilledtags.gpkg', encoding='utf-8', directed=True)



