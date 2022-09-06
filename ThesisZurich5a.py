#here i switch 2 edges at the time and I use the aa way

import sys; sys.prefix
#here i calculate variance
import os
os.getcwd()
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx

import geonetworkx as gnx
import netgraph

from operator import itemgetter
import time
from numpy import arange
from matplotlib import pyplot
import random
import math
import numpy as np
from copy import deepcopy
from itertools import chain

filepath = './data/kantstrasse.graphml'

#g = ox.load_graphml(filepath)

# here I'll add Sequence4d
g = ox.load_graphml("./data/zurichdietedD.graphml")
#####################################################

for u, v, data in g.edges(data=True):
    data['weigth'] = float(data['weigth'])

nodes, edges = ox.graph_to_gdfs(g)
gcopia = nx.MultiDiGraph(g)

# I do for primary

primarynodes = {}
primaryedges = {}

for u, v, k in gcopia.edges(data=True):
    if (k['highway'] == 'primary' or k['highway'] == 'primary_link') and k['oneway'] == True:
        primaryedges[u, v] = k
        primarynodes[u] = ''

primaryneigh = {}

for u in primarynodes:
    a = list(gcopia[u])
    for v in a:
        # print(v)
        primaryneigh[v] = ''

len(primarynodes)

primaryneigh.update(primarynodes)

len(primaryneigh)

g0 = nx.MultiDiGraph(g)
gx = nx.MultiDiGraph(g)

primaryneighedges = {}

for u, v, d in gx.edges(data=True):
    if u and v in primaryneigh and d['oneway'] == True:
        primaryneighedges[u, v] = d

print(len(primaryneighedges))

# here I do the neighlists for the other highways types

# I do for secondary

secondarynodes = {}
secondaryedges = {}

for u, v, k in gcopia.edges(data=True):
    if (k['highway'] == 'secondary' or k['highway'] == 'secondary_link') and k['oneway'] == True:
        secondaryedges[u, v] = k
        secondarynodes[u] = ''

secondaryneigh = {}

for u in secondarynodes:
    a = list(gcopia[u])
    for v in a:
        # print(v)
        secondaryneigh[v] = ''

len(secondarynodes)

secondaryneigh.update(secondarynodes)

len(secondaryneigh)

g0 = nx.MultiDiGraph(g)
gx = nx.MultiDiGraph(g)

secondaryneighedges = {}

for u, v, d in gx.edges(data=True):
    if u and v in secondaryneigh and d['oneway'] == True:
        secondaryneighedges[u, v] = d

print(len(secondaryneighedges))

# here I do the neighlists for the other highways types

# I do for tertiary

tertiarynodes = {}
tertiaryedges = {}

for u, v, k in gcopia.edges(data=True):
    if (k['highway'] == 'tertiary' or k['highway'] == 'tertiary_link') and k['oneway'] == True:
        tertiaryedges[u, v] = k
        tertiarynodes[u] = ''

tertiaryneigh = {}

for u in tertiarynodes:
    a = list(gcopia[u])
    for v in a:
        # print(v)
        tertiaryneigh[v] = ''

len(tertiarynodes)

tertiaryneigh.update(tertiarynodes)

len(tertiaryneigh)

g0 = nx.MultiDiGraph(g)
gx = nx.MultiDiGraph(g)

tertiaryneighedges = {}

for u, v, d in gx.edges(data=True):
    if u and v in tertiaryneigh and d['oneway'] == True:
        tertiaryneighedges[u, v] = d

print(len(tertiaryneighedges))

# here I do the neighlists for the other highways types

# I do for residential and living street and unclassified (assuming they are living streets)

residentialnodes = {}
residentialedges = {}

for u, v, k in gcopia.edges(data=True):
    if (k['highway'] == 'residential' or k['highway'] == 'living_street' or k['highway'] == 'unclassified') and k[
        'oneway'] == True:
        residentialedges[u, v] = k
        residentialnodes[u] = ''

residentialneigh = {}

for u in residentialnodes:
    a = list(gcopia[u])
    for v in a:
        # print(v)
        residentialneigh[v] = ''

len(residentialnodes)

residentialneigh.update(residentialnodes)

len(residentialneigh)

g0 = nx.MultiDiGraph(g)
gx = nx.MultiDiGraph(g)

residentialneighedges = {}

for u, v, d in gx.edges(data=True):
    if u and v in residentialneigh and d['oneway'] == True:
        residentialneighedges[u, v] = d

print(len(residentialneighedges))

# here SA

# if anyone wanna roll both edges (cool to determine more amp variance?)

IandIIneighedges = {}
IandIIneighedges.update(primaryneighedges)
IandIIneighedges.update(secondaryneighedges)

IandIneigh = {}
IandIneigh.update(primaryneigh)
IandIneigh.update(secondaryneigh)


# to TOUCH IG version
# NW generation to determine temperature
import igraph as ig

g0 = nx.MultiDiGraph(g)
gx = nx.MultiDiGraph(g)
n_iter = 1000
I = len(IandIIneighedges)
NFlips = 2

betlist = []
count = 0

stops = 0

# dicto = nx.edge_betweenness_centrality(g0, normalized=True, weight='weigth')

g0ig = ig.Graph.from_networkx(g0)
a = g0ig.edge_betweenness(weights='weigth')
bczero = sum(a)

# dicto = betweenness_centrality_parallel(g0)
# values = dicto.values()
# bczero = sum(values)
bcmin = bczero

vectorG = [] * 10
vectorbc = [bczero * 10] * 10
vectorbc = np.array(vectorbc)


#fliplist = []

random.seed(1311)

for it in range(0, n_iter):
    # temperature = initial_temp / (it + 1)
    # temperature = initial_temp *(1-it/n_iter)
    good = True
    while good:
        flip = False
        gx = nx.MultiDiGraph(g)
        nflip = random.randint(1, NFlips)
        chaos = random.sample(range(I), nflip)
        # print(nflip,'      ',chaos)
        #fliplist.append((nflip, chaos))
        listnode = []
        flipname = []
        flippedname = []

        for i in range(nflip):
            try:
                attributes = gx[list(IandIIneighedges)[chaos[i]][0]][list(IandIIneighedges)[chaos[i]][1]][0]

                # listnode.append([list(primaryneighedges)[chaos[i]][0], list(primaryneighedges)[chaos[i]][1]])
                gx.add_edge(list(IandIIneighedges)[chaos[i]][1], list(IandIIneighedges)[chaos[i]][0])
                gx.edges[list(IandIIneighedges)[chaos[i]][1], list(IandIIneighedges)[chaos[i]][0], 0].update(
                    attributes)
                gx.remove_edge(list(IandIIneighedges)[chaos[i]][0], list(IandIIneighedges)[chaos[i]][1])
                print(list(IandIIneighedges)[chaos[i]][0], "        ", list(IandIIneighedges)[chaos[i]][1])
                for u, v, d in gx.edges(data=True):
                    if d['name'] == attributes['name'] and (v, u) != (
                            list(IandIIneighedges)[chaos[i]][1], list(IandIIneighedges)[chaos[i]][0]) and d[
                        'oneway'] == True:
                        flipname.append((u, v))
                        flippedname.append((v, u, d))
                        # print(flipname)
                gx.add_edges_from(flippedname)
                gx.remove_edges_from(flipname)
            except KeyError:
                pass
        dtot = 1

        for u in list(IandIneigh):
            dtot = dtot * gx.in_degree(u) * gx.out_degree(u)
        if dtot != 0:
            print('puzzetta')
            if nx.is_strongly_connected(gx):
                good = False

                # obj1 = nx.edge_betweenness_centrality(gx, normalized=True, weight='weigth')
    # obj1 = betweenness_centrality_parallel(gx)
    # with igraph
    gxig = ig.Graph.from_networkx(gx)
    a = gxig.edge_betweenness(weights='weigth')
    bcx = sum(a)

    # bcx = sum(obj1.values())
    betlist.append(bcx)
    if bcx < np.min(vectorbc):
        vectorbc[vectorbc.argmax()] = bcx
        vectorG.append(gx)
    print(it)
bcsd = np.std(betlist)
print(it)

# explore temperature vs algorithm iteration for simulated annealing
import matplotlib.pyplot as plt
import math
# total iterations of algorithm
iterationsA = 1000
# initial temperature
initial_temp = 10*(bcsd)
# array of iterations from 0 to iterations - 1
iterations = [i for i in range(iterationsA)]
# temperatures for each iterations
#temperatures = [initial_temp/float(i + 1) for i in iterations]
temperaturea = [initial_temp*(1-it/iterationsA)/math.pow(float(it/iterationsA + 1),1) for it in iterations]
temperatured = [initial_temp*(1-it/iterationsA)/math.pow(float(it/iterationsA + 1),0) for it in iterations]
temperaturel = [initial_temp*(1-it/iterationsA)/math.pow(float(it/iterationsA + 1),0.5) for it in iterations]
temperaturee = [initial_temp*(1-it/iterationsA)/math.pow(float(it/iterationsA + 1),2) for it in iterations]
#temperature = [initial_temp *(1-it/n_iter) for it in iterations]
# plot iterations vs temperatures
#pyplot.plot(iterations, temperaturea, temperatured, temperaturel)#, temperaturee)
#pyplot.xlabel('Iteration')
#pyplot.ylabel('Temperature')
#pyplot.show()
print(initial_temp)

fig, ax = plt.subplots()
plt.xlabel('Iteration')
plt.ylabel('Temperature')
ax.plot(iterations, temperaturea, color = 'g')
ax.plot(iterations, temperatured, color = 'y')
ax.plot(iterations, temperaturel, color = 'b')
ax.plot(iterations, temperaturee, color = 'r')

#ax.legend(loc = 'upper left')

plt.show()


#I save to disk variance


file=open("stdZurich5a", "w")
file.write("%f\n" % bcsd)
file.close()

file=open("Zurichbclist5a", "wb")
pickle.dump(betlist, file)
file.close()

file=open("Zurichbestbcvector5a", "wb")
pickle.dump(vectorbc, file)
file.close()

file=open("ZurichGlist5a", "wb")
pickle.dump(vectorG, file)
file.close()