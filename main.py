import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt

#We use Zurich City as an example.
region_name = 'Zurich'

#sometimes you need to play with which_result parameter
#because you can receive node point, not the polygon
region = ox.geocoder.geocode_to_gdf(region_name, which_result=2)
print(region)

region.plot(figsize=(15,15))
plt.title(region_name, fontdict={'fontsize':15})
plt.grid()
plt.show()

ox.plot_graph(ox.graph_from_place('Zurich, Switzerland'))


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()

from pymapd import connect
import osmnx as ox
import geopandas as gpd

# Area of Interest Graph
G = ox.graph_from_place('Zurich, Switzerland', network_type='drive')

# Map Preview Graph
ox.plot_graph(G)

# Load Edges into GeoDataFrame
nodes, edges = ox.graph_to_gdfs(G)

# Drop Unwanted Columns
edges.drop(['oneway', 'lanes', 'maxspeed'], inplace=True, axis=1)

# Preview Data
edges

# Check Coordinate Reference System
edges.crs

#adrian meister for osmnx data
#representation of data structure, osm use polygonal algorithm uses prob matrix, attention at precision
#
