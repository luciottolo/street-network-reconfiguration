#here It will be checked how data are organized, which values are missing.
#This section could have been expanded with the calc of the relations between variables, even adding edges bc
#complete graphs present in the graphs docs
#####################################################
import sys; sys.prefix
import os
os.getcwd()
import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox

import seaborn as sns

sns.set(style="darkgrid")

from itertools import chain

#########################################
filepath = './data/zurichstronglyconnectedwbc.graphml'

G = ox.load_graphml(filepath)

#here I check what's in the columns
print(set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True))))
print(list(list(G.edges(data=True))[0][-1].keys()))

attrtobekept = ['highway', 'oneway', 'lanes', 'length', 'width', 'osmid', 'name', 'geometry', 'edge_bbcentrality']

#here I go pandas, gettin only columns of interest
nodes, edges = ox.graph_to_gdfs(G)

edges = edges[edges.columns.intersection(attrtobekept)]

g = ox.graph_from_gdfs(nodes, edges)

#I count the null values
null_df = edges.apply(lambda x: sum(x.isnull())).to_frame(name='count')
df = null_df
print(df)
print(df.to_latex(index=True))


#One by one I look into the variables

#note: if analisyng a simplified NW, to be hashed unique() as
#method create lists, and therefore gives error

#OSMid

#print(edges['osmid'].unique())
#print(edges['osmid'].nunique())
df = edges['osmid'].value_counts()
print(df)
print(df.to_latex(index=True))


df = edges.osmid.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))


#Oneway

df = pd.DataFrame(edges['oneway'].unique())
print(df)
print(df.to_latex(index=True))

print(edges['oneway'].nunique())

df = edges['oneway'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.oneway.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#Lanes

#print(edges['lanes'].unique())
#print(edges['lanes'].nunique())
df = edges['lanes'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.lanes.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))
#highway

#print(edges['highway'].unique())
#print(edges['highway'].nunique())
df = edges['highway'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.highway.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#length

print(edges['length'].unique())
print(edges['length'].nunique())
df = (edges['length'].value_counts())
print(df)
print(df.to_latex(index=True))

df = edges.length.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#name

#print(edges['name'].unique())
#print(edges['name'].nunique())
df = edges['name'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.name.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#width

#print(edges['width'].unique())
#print(edges['width'].nunique())
df = edges['width'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.width.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))


# store numerical and categorical column in two different variables. It comes handy during visualizaion.
num_col = edges._get_numeric_data().columns
cat_col = list(set(edges.columns)-set(num_col))

#plot nans
plt.plot(null_df.index, null_df['count'])
plt.xticks(null_df.index, null_df.index, rotation=45,
horizontalalignment='right')
plt.xlabel('column names')
plt.margins(0.1)
plt.show()

#when network is simplified it doesnt work, because of list stuff
'''
for i in cat_col:
   if i in ['source']:
      continue
   plt.figure(figsize=(10, 5))
   chart = sns.countplot(
   data=edges,
   x=i,
   palette='Set1'
 )
   chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
   plt.show()
'''

for i in num_col:
   if i in ['source']:
     continue
   plt.figure(figsize=(10, 5))
   chart = sns.countplot(
             data=edges,
             x=i,
             palette='Set1',
# This option plot top category of numerical values.
             order=pd.value_counts(edges[i]).iloc[:10].index
           )
   chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
   plt.show()

################################################################
#I call now the non simplified NW to get more cool graphs
#not simplified
#filepath = './data/Zurich.graphml'

#simplifiedw clean tags
filepath = './data/zurichfilledtags.graphml'

G = ox.load_graphml(filepath)

#here I check what's in the columns
print(set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True))))
print(list(list(G.edges(data=True))[0][-1].keys()))

attrtobekept = ['highway', 'oneway', 'lanes', 'length', 'width', 'osmid', 'name', 'edge_bbcentrality', 'geometry']

#here I go pandas, gettin only columns of interest
nodes, edges = ox.graph_to_gdfs(G)

edges = edges[edges.columns.intersection(attrtobekept)]

g = ox.graph_from_gdfs(nodes, edges)

#I count the null values
null_df = edges.apply(lambda x: sum(x.isnull())).to_frame(name='count')
df = null_df
print(df)
print(df.to_latex(index=True))

#One by one I look into the variables

#note: if analisyng a simplified NW, to be hashed unique() as
#method create lists, and therefore gives error

#OSMid

print(edges['osmid'].unique())
print(edges['osmid'].nunique())

df = edges['osmid'].value_counts()
print(df)
print(df.to_latex(index=True))


df = edges.osmid.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#Oneway

df = pd.DataFrame(edges['oneway'].unique())
print(df)
print(df.to_latex(index=True))

print(edges['oneway'].nunique())

df = edges['oneway'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.oneway.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#Lanes

print(edges['lanes'].unique())
print(edges['lanes'].nunique())
df = edges['lanes'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.lanes.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#highway

print(edges['highway'].unique())
print(edges['highway'].nunique())
df = edges['highway'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.highway.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#length

print(edges['length'].unique())
print(edges['length'].nunique())
df = (edges['length'].value_counts())
print(df)
print(df.to_latex(index=True))

df = edges.length.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#name

#print(edges['name'].unique())
#print(edges['name'].nunique())
df = edges['name'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.name.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))

#width

#print(edges['width'].unique())
#print(edges['width'].nunique())
df = edges['width'].value_counts()
print(df)
print(df.to_latex(index=True))

df = edges.width.apply(type).value_counts()
print(df)
print(df.to_latex(index=True))


# store numerical and categorical column in two different variables. It comes handy during visualizaion.
num_col = edges._get_numeric_data().columns
cat_col = list(set(edges.columns)-set(num_col))

#plot nans
plt.plot(null_df.index, null_df['count'])
plt.xticks(null_df.index, null_df.index, rotation=45,
horizontalalignment='right')
plt.xlabel('column names')
plt.margins(0.1)
plt.show()

#when network is simplified it doesnt work, because of list stuff

for i in cat_col:
   if i in ['source']:
      continue
   plt.figure(figsize=(10, 5))
   chart = sns.countplot(
   data=edges,
   x=i,
   palette='Set1'
 )
   chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
   plt.show()


for i in num_col:
   if i in ['source']:
     continue
   plt.figure(figsize=(10, 5))
   chart = sns.countplot(
             data=edges,
             x=i,
             palette='Set1',
# This option plot top category of numerical values.
             order=pd.value_counts(edges[i]).iloc[:10].index
           )
   chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
   plt.show()
#it gives error on geometries but it is fine, since it is automatic process


