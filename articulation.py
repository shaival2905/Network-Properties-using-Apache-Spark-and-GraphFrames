import sys
import time
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from copy import deepcopy

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)
sc.setCheckpointDir('.')

def articulations(g, usegraphframe=False):
	# Get the starting count of connected components
	connected_components = g.connectedComponents()
    initial_connected_components = connected_components.groupBy("component").count().count()
	# Default version sparkifies the connected components process
	# and serializes node iteration.
    if usegraphframe:
		# Get vertex list for serial iteration
		vertex_list = g.vertices.rdd.map(lambda x: x.id).collect()

		# For each vertex, generate a new graphframe missing that vertex
		# and calculate connected component count. Then append count to
		# the output
        articulation_list = []
        for vertex in vertex_list:
            next_vertices = g.vertices.filter(functions.col("id") != functions.lit(vertex))
            next_edges = g.edges.filter(~(functions.col("src")==functions.lit(vertex)) & ~(functions.col("dst") == functions.lit(vertex)))
            g2 = GraphFrame(next_vertices, next_edges)
            next_connected_components = g2.connectedComponents().groupBy("component").count().count()
            if next_connected_components > initial_connected_components:
                articulation_list.append((vertex, 1))
            else:
                articulation_list.append((vertex, 0))

        return sqlContext.createDataFrame(articulation_list,['id','articulation'])
	# Non-default version sparkifies node iteration and uses networkx
	# for connected components count.
    else:
        nx_graph = nx.Graph()
        def getVertexId(vertex):
            return vertex.id
        nx_graph.add_nodes_from(g.vertices.rdd.map(lambda vertex: getVertexId(vertex)).collect())
        def getEdgeTuple(edge):
            return (edge.src, edge.dst)
        nx_graph.add_edges_from(g.edges.rdd.map(lambda edge: getEdgeTuple(edge)).collect())

        articulation_list = []
        node_list = list(nx_graph.nodes)
        for i in range(len(node_list)):
            list_subset = node_list[:i] + node_list[i+1:]
            sub_graph = nx_graph.subgraph(list_subset)
            next_connected_components = nx.number_connected_components(sub_graph)
            if next_connected_components > initial_connected_components:
                articulation_list.append((node_list[i],1))
            else:
                articulation_list.append((node_list[i],0))

        return sqlContext.createDataFrame(articulation_list,['id','articulation'])


filename = sys.argv[1]
lines = sc.textFile(filename)

pairs = lines.map(lambda s: s.split(","))
e = sqlContext.createDataFrame(pairs,['src','dst'])
e = e.unionAll(e.selectExpr('src as dst','dst as src')).distinct() # Ensure undirectedness

# Extract all endpoints from input file and make a single column frame.
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()

# Create graphframe from the vertices and edges.
g = GraphFrame(v,e)

#Runtime approximately 5 minutes
print("---------------------------")
print("Processing graph using Spark iteration over nodes and serial (networkx) connectedness calculations")
init = time.time()
df = articulations(g, False)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
articulations_df = df.filter('articulation = 1')
articulations_df.show(truncate=False)
articulations_df.toPandas().to_csv("articulations_out.csv")
print("---------------------------")

#Runtime for below is more than 2 hours
print("Processing graph using serial iteration over nodes and GraphFrame connectedness calculations")
init = time.time()
df = articulations(g, True)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
