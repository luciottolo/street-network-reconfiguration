import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import pickle
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
import igraph as ig

filepath = './data/kantstrasse.graphml'
#g = ox.load_graphml(filepath)

# here I'll add Sequence4d
g = ox.load_graphml("./data/zurichdietedD.graphml")
#####################################################
##I upload saved to disk files

file = open("stdZurich5", "r")
bcsd = file.readline()
file.close()

file = open("Zurichbclist5", "rb")
betlist = pickle.load(file)
file.close()

file = open("Zurichbestbcvector5", "rb")
vectorbc = pickle.load(file)
file.close()

f = open("ZurichGlist5", "rb")
vectorG = pickle.load(f)
file.close()


########################################################
initial_temp = 10*(float(bcsd))
print(type(initial_temp))

for u, v, data in g.edges(data=True):
    data['weigth'] = float(data['weigth'])
    data['length'] = float(data['length'])

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

#to be continued
#upload file w sd and w 10bestgraph w bc  AND MAYBE all solutions to make stats
#insert SA
###########################################################################################################################

# to TOUCH IGRAPH VERSION
# primary


g0 = nx.MultiDiGraph(g)
gx = nx.MultiDiGraph(g)
n_iter = 1000
I = len(primaryneighedges)
NFlips = 3

count = 0

stops = 0

g0ig = ig.Graph.from_networkx(g0)
a = g0ig.edge_betweenness(weights='length')
bczero = sum(a)
bcinitial = bczero
print('starting bc:', bczero)

# dicto = nx.edge_betweenness_centrality(g0, normalized=True, weight='weigth')
# values = dicto.values()
# bczero = sum(values)
bcmin = bczero

# initial_temp = bczero/4
# initial_temp

# fliplist = []
# chancearray = []
# for i in primaryneighedges:
#    chancearray.append(0)
# vectorflip = []


random.seed(1311)

for it in range(0, n_iter):
    # temperature = initial_temp / (it + 1)
    #temperature = initial_temp * (1 - it / n_iter)
    temperature = initial_temp * (1 - it / n_iter) / math.pow(float(it / n_iter + 1), 1)
    good = True
    while good:
        flip = False
        gx = nx.MultiDiGraph(g)
        nflip = random.randint(1, NFlips)
        chaos = random.sample(range(I), nflip)
        # print(nflip,'      ',chaos)
        #fliplist.append(chaos)
        listnode = []

        for i in range(nflip):
            attributes = gx[list(primaryneighedges)[chaos[i]][0]][list(primaryneighedges)[chaos[i]][1]][0]

            listnode.append([list(primaryneighedges)[chaos[i]][0], list(primaryneighedges)[chaos[i]][1]])
            gx.add_edge(list(primaryneighedges)[chaos[i]][1], list(primaryneighedges)[chaos[i]][0])
            gx.edges[list(primaryneighedges)[chaos[i]][1], list(primaryneighedges)[chaos[i]][0], 0].update(attributes)
            gx.remove_edge(list(primaryneighedges)[chaos[i]][0], list(primaryneighedges)[chaos[i]][1])
        dtot = 1
        count = count + 1
        # print(count)
        for u in list(primaryneigh):
            dtot = dtot * gx.in_degree(u) * gx.out_degree(u)
        if dtot != 0:
            print('puzzetta')
            if nx.is_strongly_connected(gx):
                good = False

                # obj1 = nx.edge_betweenness_centrality(gx, normalized=True, weight='weigth')
    # with igraph
    gxig = ig.Graph.from_networkx(gx)
    a = gxig.edge_betweenness(weights='length')
    bcx = sum(a)
    betlist.append(bcx)
    # bcx = sum(obj1.values())
    print(bcx, bczero, temperature)
    if bcx < bczero:
        g0 = nx.MultiDiGraph(gx)
        bczero = bcx
        bcmin = bczero
        if bcx < np.min(vectorbc):
            vectorbc[vectorbc.argmax()] = bcx
            vectorG.append(gx)
            # for w in chaos:
            #   print('questo e w', w)

            """if chancearray[w]==0:
                    chancearray[w]=1
                else:
                    chancearray[w]=0
            vectorflip.append(chancearray)
            """

    else:
        if math.exp(-(bcx - bczero) / temperature) > random.uniform(0, 1):
            g0 = nx.MultiDiGraph(gx)
            print(math.exp(-(bcx - bczero) / temperature))
            bczero = bcx

            # print(g0.edges,gx.edges)
        else:
            pass
    if bcmin == bczero:
        stops = stops + 1
    else:
        stops = 0
    if stops > 50:
        break
print(it)

print(bczero, "       ", bcmin)

######################################################3
#secondary

n_iter = 1000
I = len(secondaryneighedges)
NFlips = 3

count = 0

stops = 0

g0ig = ig.Graph.from_networkx(g0)
a = g0ig.edge_betweenness(weights='length')
bczero = sum(a)

# dicto = nx.edge_betweenness_centrality(g0, normalized=True, weight='weigth')
# values = dicto.values()
# bczero = sum(values)
bcmin = bczero




random.seed(1311)

for it in range(0, n_iter):
    # temperature = initial_temp / (it + 1)
    #temperature = initial_temp * (1 - it / n_iter)
    temperature = initial_temp * (1 - it / n_iter) / math.pow(float(it / n_iter + 1), 1)
    good = True
    while good:
        flip = False
        gx = nx.MultiDiGraph(g)
        nflip = random.randint(1, NFlips)
        chaos = random.sample(range(I), nflip)
        # print(nflip,'      ',chaos)
        #fliplist.append(chaos)
        listnode = []

        for i in range(nflip):
            attributes = gx[list(secondaryneighedges)[chaos[i]][0]][list(secondaryneighedges)[chaos[i]][1]][0]

            listnode.append([list(secondaryneighedges)[chaos[i]][0], list(secondaryneighedges)[chaos[i]][1]])
            gx.add_edge(list(secondaryneighedges)[chaos[i]][1], list(secondaryneighedges)[chaos[i]][0])
            gx.edges[list(secondaryneighedges)[chaos[i]][1], list(secondaryneighedges)[chaos[i]][0], 0].update(attributes)
            gx.remove_edge(list(secondaryneighedges)[chaos[i]][0], list(secondaryneighedges)[chaos[i]][1])
        dtot = 1
        count = count + 1
        # print(count)
        for u in list(secondaryneigh):
            dtot = dtot * gx.in_degree(u) * gx.out_degree(u)
        if dtot != 0:
            #print('puzzetta')
            if nx.is_strongly_connected(gx):
                good = False

                # obj1 = nx.edge_betweenness_centrality(gx, normalized=True, weight='weigth')
    # with igraph
    gxig = ig.Graph.from_networkx(gx)
    a = gxig.edge_betweenness(weights='length')
    bcx = sum(a)
    betlist.append(bcx)
    # bcx = sum(obj1.values())
    print(bcx, bczero, temperature)
    if bcx < bczero:
        g0 = nx.MultiDiGraph(gx)
        bczero = bcx
        bcmin = bczero
        if bcx < np.min(vectorbc):
            vectorbc[vectorbc.argmax()] = bcx
            vectorG.append(gx)
            # for w in chaos:
            #   print('questo e w', w)

            """if chancearray[w]==0:
                    chancearray[w]=1
                else:
                    chancearray[w]=0
            vectorflip.append(chancearray)
            """

    else:
        if math.exp(-(bcx - bczero) / temperature) > random.uniform(0, 1):
            g0 = nx.MultiDiGraph(gx)
            print(math.exp(-(bcx - bczero) / temperature))
            bczero = bcx

            # print(g0.edges,gx.edges)
        else:
            pass
    if bcmin == bczero:
        stops = stops + 1
    else:
        stops = 0
    if stops > 50:
        break
print(it)

print(bczero, "       ", bcmin)

######################################
#tertiary

#tertiary

n_iter = 1000
I = len(tertiaryneighedges)
NFlips = 3

count = 0

stops = 0

g0ig = ig.Graph.from_networkx(g0)
a = g0ig.edge_betweenness(weights='length')
bczero = sum(a)

# dicto = nx.edge_betweenness_centrality(g0, normalized=True, weight='weigth')
# values = dicto.values()
# bczero = sum(values)
bcmin = bczero




random.seed(1311)

for it in range(0, n_iter):
    # temperature = initial_temp / (it + 1)
    #temperature = initial_temp * (1 - it / n_iter)
    temperature = initial_temp * (1 - it / n_iter) / math.pow(float(it / n_iter + 1), 1)
    good = True
    while good:
        flip = False
        gx = nx.MultiDiGraph(g)
        nflip = random.randint(1, NFlips)
        chaos = random.sample(range(I), nflip)
        # print(nflip,'      ',chaos)
        #fliplist.append(chaos)
        listnode = []

        for i in range(nflip):
            attributes = gx[list(tertiaryneighedges)[chaos[i]][0]][list(tertiaryneighedges)[chaos[i]][1]][0]

            listnode.append([list(tertiaryneighedges)[chaos[i]][0], list(tertiaryneighedges)[chaos[i]][1]])
            gx.add_edge(list(tertiaryneighedges)[chaos[i]][1], list(tertiaryneighedges)[chaos[i]][0])
            gx.edges[list(tertiaryneighedges)[chaos[i]][1], list(tertiaryneighedges)[chaos[i]][0], 0].update(attributes)
            gx.remove_edge(list(tertiaryneighedges)[chaos[i]][0], list(tertiaryneighedges)[chaos[i]][1])
        dtot = 1
        count = count + 1
        # print(count)
        for u in list(tertiaryneigh):
            dtot = dtot * gx.in_degree(u) * gx.out_degree(u)
        if dtot != 0:
            #print('puzzetta')
            if nx.is_strongly_connected(gx):
                good = False

                # obj1 = nx.edge_betweenness_centrality(gx, normalized=True, weight='weigth')
    # with igraph
    gxig = ig.Graph.from_networkx(gx)
    a = gxig.edge_betweenness(weights='length')
    bcx = sum(a)
    betlist.append(bcx)
    # bcx = sum(obj1.values())
    print(bcx, bczero, temperature)
    if bcx < bczero:
        g0 = nx.MultiDiGraph(gx)
        bczero = bcx
        bcmin = bczero
        if bcx < np.min(vectorbc):
            vectorbc[vectorbc.argmax()] = bcx
            vectorG.append(gx)
            # for w in chaos:
            #   print('questo e w', w)

            """if chancearray[w]==0:
                    chancearray[w]=1
                else:
                    chancearray[w]=0
            vectorflip.append(chancearray)
            """

    else:
        if math.exp(-(bcx - bczero) / temperature) > random.uniform(0, 1):
            g0 = nx.MultiDiGraph(gx)
            print(math.exp(-(bcx - bczero) / temperature))
            bczero = bcx

            # print(g0.edges,gx.edges)
        else:
            pass
    if bcmin == bczero:
        stops = stops + 1
    else:
        stops = 0
    if stops > 50:
        break
print(it)

print(bczero, "       ", bcmin)

#################
#residential


n_iter = 1000
I = len(residentialneighedges)
NFlips = 3

count = 0

stops = 0

g0ig = ig.Graph.from_networkx(g0)
a = g0ig.edge_betweenness(weights='length')
bczero = sum(a)

# dicto = nx.edge_betweenness_centrality(g0, normalized=True, weight='weigth')
# values = dicto.values()
# bczero = sum(values)
bcmin = bczero




random.seed(1311)

for it in range(0, n_iter):
    # temperature = initial_temp / (it + 1)
    #temperature = initial_temp * (1 - it / n_iter)
    temperature = initial_temp * (1 - it / n_iter) / math.pow(float(it / n_iter + 1), 1)
    good = True
    while good:
        flip = False
        gx = nx.MultiDiGraph(g)
        nflip = random.randint(1, NFlips)
        chaos = random.sample(range(I), nflip)
        # print(nflip,'      ',chaos)
        #fliplist.append(chaos)
        listnode = []

        for i in range(nflip):
            attributes = gx[list(residentialneighedges)[chaos[i]][0]][list(residentialneighedges)[chaos[i]][1]][0]

            listnode.append([list(residentialneighedges)[chaos[i]][0], list(residentialneighedges)[chaos[i]][1]])
            gx.add_edge(list(residentialneighedges)[chaos[i]][1], list(residentialneighedges)[chaos[i]][0])
            gx.edges[list(residentialneighedges)[chaos[i]][1], list(residentialneighedges)[chaos[i]][0], 0].update(attributes)
            gx.remove_edge(list(residentialneighedges)[chaos[i]][0], list(residentialneighedges)[chaos[i]][1])
        dtot = 1
        count = count + 1
        # print(count)
        for u in list(residentialneigh):
            dtot = dtot * gx.in_degree(u) * gx.out_degree(u)
        if dtot != 0:
            #print('puzzetta')
            if nx.is_strongly_connected(gx):
                good = False

                # obj1 = nx.edge_betweenness_centrality(gx, normalized=True, weight='weigth')
    # with igraph
    gxig = ig.Graph.from_networkx(gx)
    a = gxig.edge_betweenness(weights='length')
    bcx = sum(a)
    betlist.append(bcx)
    # bcx = sum(obj1.values())
    print(bcx, bczero, temperature)
    if bcx < bczero:
        g0 = nx.MultiDiGraph(gx)
        bczero = bcx
        bcmin = bczero
        if bcx < np.min(vectorbc):
            vectorbc[vectorbc.argmax()] = bcx
            vectorG.append(gx)
            # for w in chaos:
            #   print('questo e w', w)

            """if chancearray[w]==0:
                    chancearray[w]=1
                else:
                    chancearray[w]=0
            vectorflip.append(chancearray)
            """

    else:
        if math.exp(-(bcx - bczero) / temperature) > random.uniform(0, 1):
            g0 = nx.MultiDiGraph(gx)
            print(math.exp(-(bcx - bczero) / temperature))
            bczero = bcx

            # print(g0.edges,gx.edges)
        else:
            pass
    if bcmin == bczero:
        stops = stops + 1
    else:
        stops = 0
    if stops > 50:
        break
print(it)

print(bczero, "       ", bcmin)

#get min graph depending on min bc
#questa e una bella roba non te la scorda

Gminimum = nx.MultiDiGraph(vectorG[np.argmin(vectorbc)])

bcfinal = np.min(vectorbc)

#to get position
#np.argmin(vectorbc)

print('bc % change:  ',(bcinitial-bcfinal)*100/bcinitial)

##I save some s#$%t to disk

file=open("bcfinalA", "w")
file.write("%f\n" % bcfinal)
file.close()

file=open("bclistA", "wb")
pickle.dump(betlist, file)
file.close()

file=open("GminimumA", "wb")
pickle.dump(Gminimum, file)
file.close()


