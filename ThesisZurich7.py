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

#####################################################
g = ox.load_graphml("./data/zurichdietedD.graphml")

gfilled = ox.load_graphml("./data/zurichfilledtags.graphml")

for u,v,data in g.edges(data=True):
    data['weigth']=float(data['weigth'])
    data['length']=float(data['length'])

for u,v,data in gfilled.edges(data=True):
    data['width']=float(data['width'])
    data['length']=float(data['length'])

weigth = {(u,v,key) : data['length']/data['width'] for u, v, key, data in gfilled.edges(keys=True, data=True)}

nx.set_edge_attributes(gfilled, weigth, 'weigth')

gi = ig.Graph.from_networkx(g)

a = gi.edge_betweenness(weights = 'length')

bc0 = sum(a)
########################################################
##I upload saved to disk files
#A

file = open("bcfinalA", "r")
bcA = file.readline()
file.close()

file = open("bclistA", "rb")
bclistA = pickle.load(file)
file.close()

file = open("GminimumA", "rb")
GminimumA = pickle.load(file)
file.close()

#AA

file = open("bcfinalAA", "r")
bcAA = file.readline()
file.close()

file = open("bclistAA", "rb")
bclistAA = pickle.load(file)
file.close()

file = open("GminimumAA", "rb")
GminimumAA = pickle.load(file)
file.close()

#B

file = open("bcfinalB", "r")
bcB = file.readline()
file.close()

file = open("bclistB", "rb")
bclistB = pickle.load(file)
file.close()

file = open("GminimumB", "rb")
GminimumB = pickle.load(file)
file.close()

#BB

file = open("bcfinalBB", "r")
bcBB = file.readline()
file.close()

file = open("bclistBB", "rb")
bclistBB = pickle.load(file)
file.close()

file = open("GminimumBB", "rb")
GminimumBB = pickle.load(file)
file.close()

#C

file = open("bcfinalC", "r")
bcC = file.readline()
file.close()

file = open("bclistC", "rb")
bclistC = pickle.load(file)
file.close()

file = open("GminimumC", "rb")
GminimumC = pickle.load(file)
file.close()

#CC

file = open("bcfinalCC", "r")
bcCC = file.readline()
file.close()

file = open("bclistCC", "rb")
bclistCC = pickle.load(file)
file.close()

file = open("GminimumCC", "rb")
GminimumCC = pickle.load(file)
file.close()

#D

file = open("bcfinalD", "r")
bcD = file.readline()
file.close()

file = open("bclistD", "rb")
bclistD = pickle.load(file)
file.close()

file = open("GminimumD", "rb")
GminimumD = pickle.load(file)
file.close()

#DD

file = open("bcfinalDD", "r")
bcDD = file.readline()
file.close()

file = open("bclistDD", "rb")
bclistDD = pickle.load(file)
file.close()

file = open("GminimumDD", "rb")
GminimumDD = pickle.load(file)
file.close()

#E

file = open("bcfinalE", "r")
bcE = file.readline()
file.close()

file = open("bclistE", "rb")
bclistE = pickle.load(file)
file.close()

file = open("GminimumE", "rb")
GminimumE = pickle.load(file)
file.close()

#EE

file = open("bcfinalEE", "r")
bcEE = file.readline()
file.close()

file = open("bclistEE", "rb")
bclistEE = pickle.load(file)
file.close()

file = open("GminimumEE", "rb")
GminimumEE = pickle.load(file)
file.close()

#F

file = open("bcfinalF", "r")
bcF = file.readline()
file.close()

file = open("bclistF", "rb")
bclistF = pickle.load(file)
file.close()

file = open("GminimumF", "rb")
GminimumF = pickle.load(file)
file.close()

#FF

file = open("bcfinalFF", "r")
bcFF = file.readline()
file.close()

file = open("bclistFF", "rb")
bclistFF = pickle.load(file)
file.close()

file = open("GminimumFF", "rb")
GminimumFF = pickle.load(file)
file.close()

#G

file = open("bcfinalG", "r")
bcG = file.readline()
file.close()

file = open("bclistG", "rb")
bclistG = pickle.load(file)
file.close()

file = open("GminimumG", "rb")
GminimumG = pickle.load(file)
file.close()

#GG

file = open("bcfinalGG", "r")
bcGG = file.readline()
file.close()

file = open("bclistGG", "rb")
bclistGG = pickle.load(file)
file.close()

file = open("GminimumGG", "rb")
GminimumGG = pickle.load(file)
file.close()

#H

file = open("bcfinalH", "r")
bcH = file.readline()
file.close()

file = open("bclistH", "rb")
bclistH = pickle.load(file)
file.close()

file = open("GminimumH", "rb")
GminimumH = pickle.load(file)
file.close()

#HH

file = open("bcfinalHH", "r")
bcHH = file.readline()
file.close()

file = open("bclistHH", "rb")
bclistHH = pickle.load(file)
file.close()

file = open("GminimumHH", "rb")
GminimumHH = pickle.load(file)
file.close()

#I

file = open("bcfinalI", "r")
bcI = file.readline()
file.close()

file = open("bclistI", "rb")
bclistI = pickle.load(file)
file.close()

file = open("GminimumI", "rb")
GminimumI = pickle.load(file)
file.close()

#II

file = open("bcfinalII", "r")
bcII = file.readline()
file.close()

file = open("bclistII", "rb")
bclistII = pickle.load(file)
file.close()

file = open("GminimumII", "rb")
GminimumII = pickle.load(file)
file.close()

#L

file = open("bcfinalL", "r")
bcL = file.readline()
file.close()

file = open("bclistL", "rb")
bclistL = pickle.load(file)
file.close()

file = open("GminimumL", "rb")
GminimumL = pickle.load(file)
file.close()

#LL

file = open("bcfinalLL", "r")
bcLL = file.readline()
file.close()

file = open("bclistLL", "rb")
bclistLL = pickle.load(file)
file.close()

file = open("GminimumLL", "rb")
GminimumLL = pickle.load(file)
file.close()
'''
#M

file = open("bcfinalM", "r")
bcM = file.readline()
file.close()

file = open("bclistM", "rb")
bclistM = pickle.load(file)
file.close()

file = open("GminimumM", "rb")
GminimumM = pickle.load(file)
file.close()

#MM

file = open("bcfinalMM", "r")
bcMM = file.readline()
file.close()

file = open("bclistMM", "rb")
bclistMM = pickle.load(file)
file.close()

file = open("GminimumMM", "rb")
GminimumMM = pickle.load(file)
file.close()
'''


###########################################################

bclisttot = bclistA + bclistAA + bclistB + bclistBB + bclistC + bclistCC + bclistD + bclistDD + bclistE + bclistEE + bclistF + bclistFF + bclistG + bclistGG + bclistH + bclistHH + bclistI + bclistII + bclistL + bclistLL #+ bclistM + bclistMM

bcfinalTOT = []
bcfinalTOT.append(bcA)
bcfinalTOT.append(bcAA)
bcfinalTOT.append(bcB)
bcfinalTOT.append(bcBB)
bcfinalTOT.append(bcC)
bcfinalTOT.append(bcCC)
bcfinalTOT.append(bcD)
bcfinalTOT.append(bcDD)
bcfinalTOT.append(bcE)
bcfinalTOT.append(bcEE)
bcfinalTOT.append(bcF)
bcfinalTOT.append(bcFF)
bcfinalTOT.append(bcG)
bcfinalTOT.append(bcGG)
bcfinalTOT.append(bcH)
bcfinalTOT.append(bcHH)
bcfinalTOT.append(bcI)
bcfinalTOT.append(bcII)
bcfinalTOT.append(bcL)
bcfinalTOT.append(bcLL)
#bcfinalTOT.append(bcM)
#bcfinalTOT.append(bcMM)


GfinalTOT = []
GfinalTOT.append(GminimumA)
GfinalTOT.append(GminimumAA)
GfinalTOT.append(GminimumB)
GfinalTOT.append(GminimumBB)
GfinalTOT.append(GminimumC)
GfinalTOT.append(GminimumCC)
GfinalTOT.append(GminimumD)
GfinalTOT.append(GminimumDD)
GfinalTOT.append(GminimumE)
GfinalTOT.append(GminimumEE)
GfinalTOT.append(GminimumF)
GfinalTOT.append(GminimumFF)
GfinalTOT.append(GminimumG)
GfinalTOT.append(GminimumGG)
GfinalTOT.append(GminimumH)
GfinalTOT.append(GminimumHH)
GfinalTOT.append(GminimumI)
GfinalTOT.append(GminimumII)
GfinalTOT.append(GminimumL)
GfinalTOT.append(GminimumLL)
#GfinalTOT.append(GminimumM)
#GfinalTOT.append(GminimumMM)

#GET MINIMUM CONFIG
bcfinalTOT = [float(x) for x in bcfinalTOT]

a = np.argmin(bcfinalTOT)
print(a)

GminimumABSOL = nx.MultiDiGraph(GfinalTOT[np.argmin(bcfinalTOT)])




bcminABSOL = np.min(bcfinalTOT)

#fare statistica sulla lista delle bc
#fare differenza di edges su grafico minimo assoluto e g0

print('bc0 : ', bc0, 'bcend  :', bcminABSOL)
print('bc % change:  ',(bc0-bcminABSOL)*100/bc0)

# identify the nodes/edges that were deleted/added
nodes_del = g.nodes - GminimumABSOL.nodes
nodes_add = GminimumABSOL.nodes - g.nodes
edges_del = g.edges - GminimumABSOL.edges
edges_add = GminimumABSOL.edges - g.edges


print(edges_add)
print(edges_del)
print(nodes_add)

###i ADD TRAVEL DISTANCES
#initial

random.seed(1311)

pathTOT = []
distTOT = []
n = 250

sources = np.random.choice(g.nodes, size=n, replace=True)
destinations = sources
#destinations = np.random.choice(g.nodes, size=n, replace=True)

#I calclate for filledtags
disfil = []
pathsfil = []
for s in sources:
    for d in destinations:
        disfil.append(nx.shortest_path_length(gfilled, source=s, target=d, weight='length'))
        pathsfil.append(nx.shortest_path(gfilled, source=s, target=d, weight='length'))

print(disfil)
print(pathsfil)
distTOTFil = sum(disfil)

print('TOTAL DISTANCE filled    ',distTOTFil)

#i calculate for dieted 4d
dis = []
paths = []
for s in sources:
    for d in destinations:
        dis.append(nx.shortest_path_length(g, source=s, target=d, weight='length'))
        paths.append(nx.shortest_path(g, source=s, target=d, weight='length'))

print(dis)
print(paths)
distTOT = sum(dis)
print('TOTAL DISTANCE diet    ',distTOT)

#########final


# calculate shortest-path routes using random origin-destination pairs
g0 = nx.MultiDiGraph(GminimumABSOL)

disF = []
pathsF = []
for s in sources:
    for d in destinations:
        disF.append(nx.shortest_path_length(g0, source=s, target=d,weight='length'))
        pathsF.append(nx.shortest_path(g0, source=s, target=d,weight='length'))

print(disF)
print(paths)
print(pathsF)
distTOTF = sum(disF)

print('before diet    ', distTOTFil)
print('after diet    ', distTOT)
print('after reconfig   ', distTOTF)
print('difference of distances :    ')
print('delta diet     ', distTOT-distTOTFil)
print('delta reconfig   ', distTOTF-distTOT)

print('% diet', (distTOT-distTOTFil)/distTOTFil)
print('% config  ', (distTOTF-distTOT)/distTOT)

nc = ["r" if node in destinations else "w" for node in g.nodes]
ns = [40 if node in destinations else 1 for node in g.nodes]
fig, ax = ox.plot_graph(g, node_size=ns, node_color=nc, node_zorder=2)

