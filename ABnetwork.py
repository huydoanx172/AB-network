from sklearn.cluster import KMeans
from kmeans import distance
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from util import PriorityQueue, cycle
import sys

NUM_POINTS = 100
CLUSTERS = 5
random.seed(420)

A = []
B = []
terminals = []
edges = {}

class Terminal:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.deg = 0
    def getCoords(self):
        return (self.x, self.y)

    def __eq__(self, __o: object) -> bool:
        return self.x == __o.x and self.y == __o.y
    def __hash__(self) -> int:
        return hash((self.x, self.y))
    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
    def __ge__(self, __o) -> bool:
        return self.__hash__() >= __o.__hash__()
    def __le__(self, __o) -> bool:
        return self.__hash__() <= __o.__hash__()
    def __lt__(self, __o) -> bool:
        return self.__hash__() < __o.__hash__()
    def __gt__(self, __o) -> bool:
        return self.__hash__() > __o.__hash__()

class Source(Terminal):
    def __init__(self, x, y) -> None:
        super().__init__(x, y)

class Sink(Terminal):
    def __init__(self, x, y) -> None:
        super().__init__(x, y)

def prims(points):
    """
    Take in a list of points in the Euclidean plane and return the minimum spanning
    tree as a list of points
    """
    distances = {}
    visited = []
    MST = []

    for i in range(len(points)):
        distances[points[i]] = []
        for j in range(len(points)):
            if i == j:
                distances[points[i]].append((points[i], float("inf")))
            else:
                distances[points[i]].append((points[j], distance(points[i].getCoords(), points[j].getCoords())))

    frontier = PriorityQueue()
    for edge in distances[points[0]]:
        frontier.push((points[0], edge[0]), edge[1])
    visited.append(points[0])

    while not frontier.isEmpty() and len(visited) != len(points):
        potentialEdge = frontier.pop()
        if potentialEdge[1] not in visited:
            newPoint = potentialEdge[1]
            visited.append(newPoint)
            MST.append((potentialEdge[0], potentialEdge[1]))
            for edge in distances[newPoint]:
                # print("node0 =", potentialEdge[0])
                # print("edge[0] =", edge[0])
                # print("edge[1] =", edge[1])
                frontier.push((newPoint, edge[0]), edge[1])

    if len(visited) != len(points):
        print("not all nodes have been visited in prims", file=sys.stderr)
        return None
    return MST

# Initialize terminals
for i in range(NUM_POINTS):
    source = Source(random.randint(10, 500), random.randint(10, 500))
    A.append(source)
    edges[source] = []
    sink = Sink(random.randint(10, 500), random.randint(10, 500))
    B.append(sink)
    edges[source] = []
    edges[sink] = []
    terminals.append(source)
    terminals.append(sink)

# Cluster the terminals
kmeanAll = KMeans(n_clusters=CLUSTERS*2+1, n_init=10)
kmeanSources = KMeans(n_clusters=CLUSTERS, n_init=10)
kmeanSinks = KMeans(n_clusters=CLUSTERS, n_init=10)
# Turn the middle cluster into a cycle. Cycle currently created just by a greedy
# algorithm
data = np.array([(terminal.x, terminal.y) for terminal in terminals])
kmeanAll.fit(data)

bestDist = float("inf")
centerCentroid = None
centralIndex = 0
for i in range(len(kmeanAll.cluster_centers_)):
    totalDist = 0
    # print(kmeanAll.cluster_centers_[i])
    for neighbor in kmeanAll.cluster_centers_:
        totalDist += distance(kmeanAll.cluster_centers_[i], neighbor)
    
    if totalDist < bestDist:
        bestDist = totalDist
        centerCentroid = kmeanAll.cluster_centers_[i]
        centralIndex = i

# print(kmeanAll.cluster_centers_[centralIndex])
# print("central index =", centralIndex)

cycleNodes = []
# print("labels:")
# print(kmeanAll.labels_)
for i in range(len(kmeanAll.labels_)):
    if kmeanAll.labels_[i] == centralIndex:
        cycleNodes.append(terminals[i])

# Create a cycle


for node in cycleNodes:
    # Remove from the ones that will be clustered
    if node in A:
        A.remove(node)
    else:
        B.remove(node)

# Cluster the sources and sinks
data = np.array([(terminal.x, terminal.y) for terminal in A])
kmeanSources.fit(data)
data = np.array([(terminal.x, terminal.y) for terminal in B])
kmeanSinks.fit(data)

for i in range(CLUSTERS):
    # Create an MST between points of the same type in the same cluster
    ACluster = []
    BCluster = []
    for j in range(len(kmeanSources.labels_)):
        if kmeanSources.labels_[j] == i:
            ACluster.append(A[j])
    for j in range(len(kmeanSinks.labels_)):
        if kmeanSinks.labels_[j] == i:
            BCluster.append(B[j])

    # print(ACluster)
    # print(BCluster)
    edges_to_add = prims(ACluster)
    for edge in edges_to_add:
        edges[edge[0]].append(edge[1])

    edges_to_add = prims(BCluster)
    for edge in edges_to_add:
        edges[edge[0]].append(edge[1])

# print("edges:")
# print(edges)

# Visualization
# Draw points
terminals_x = np.array([terminal.x for terminal in terminals])
terminals_y = np.array([terminal.y for terminal in terminals])
plt.scatter(terminals_x, terminals_y, s=[10 for i in range(len(terminals_x))])

# Draw lines
for tail in edges:
    for head in edges[tail]:
        if tail in A:
            x = (tail.x, head.x)
            y = (tail.y, head.y)
            plt.plot(x, y, color="green")
        else:
            x = (tail.x, head.x)
            y = (tail.y, head.y)
            plt.plot(x, y, color="red")

plt.show()