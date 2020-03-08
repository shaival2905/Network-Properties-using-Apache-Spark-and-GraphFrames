from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from pyspark.sql.functions import explode

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)

def closeness(g):

	# Get list of vertices. We'll generate all the shortest paths at
	# once using this list.
    vertices = g.vertices.rdd.map(lambda x: x.id).collect()

	# first get all the path lengths.
    path_lengths = g.shortestPaths(landmarks=vertices)
    path_lengths = path_lengths.withColumn("distances", path_lengths["distances"].cast('map<string,float>'))

	# Break up the map and group by ID for summing
	# Sum by ID
	# Get the inverses and generate desired dataframe.
    def getInvDf(path_length):
        return (path_length.id,float(1/sum(path_length.distances.values())))
    closeness_df = path_lengths.rdd.map(lambda path_length: getInvDf(path_length))

    return sqlContext.createDataFrame(closeness_df, ['id','closeness'])

print("Reading in graph for problem 2.")
graph = sc.parallelize([('A','B'),('A','C'),('A','D'),
	('B','A'),('B','C'),('B','D'),('B','E'),
	('C','A'),('C','B'),('C','D'),('C','F'),('C','H'),
	('D','A'),('D','B'),('D','C'),('D','E'),('D','F'),('D','G'),
	('E','B'),('E','D'),('E','F'),('E','G'),
	('F','C'),('F','D'),('F','E'),('F','G'),('F','H'),
	('G','D'),('G','E'),('G','F'),
	('H','C'),('H','F'),('H','I'),
	('I','H'),('I','J'),
	('J','I')])

e = sqlContext.createDataFrame(graph,['src','dst'])
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()
print("Generating GraphFrame.")
g = GraphFrame(v,e)
print("Calculating closeness.")
closeness_sorted = closeness(g).sort('closeness',ascending=False)
closeness_sorted.show()
closeness_sorted.toPandas().to_csv("centrality_out.csv")
