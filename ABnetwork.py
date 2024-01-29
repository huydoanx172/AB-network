from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from util import PriorityQueue, findMinCycle, getEuclideanDistances, distance, findMinEdgeBetweenClusters, totalCost
from python_tsp.exact import solve_tsp_dynamic_programming
import sys

NUM_POINTS = 60
CLUSTERS = 4
random.seed(10)

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
    distances = getEuclideanDistances(points)
    visited = []
    MST = []

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
                frontier.push((newPoint, edge[0]), edge[1])

    if len(visited) != len(points):
        print("not all nodes have been visited in prims", file=sys.stderr)
        return None
    return MST

A = []
B = []
terminals = []
edges = {}

# Initialize terminals
for i in range(NUM_POINTS):
    # Randomly appoint whether a point is a source or a sink
    isSource = random.choice([True, False])
    if isSource:
        newNode = Source(random.randint(10, 500), random.randint(10, 500))
        A.append(newNode)
    else: 
        newNode = Sink(random.randint(10, 500), random.randint(10, 500))
        B.append(newNode)
    edges[newNode] = []
    terminals.append(newNode)

# Cluster the terminals
kmeanAll = KMeans(n_clusters=CLUSTERS*2+1, n_init=10)
kmeanSources = KMeans(n_clusters=CLUSTERS, n_init=10)
kmeanSinks = KMeans(n_clusters=CLUSTERS, n_init=10)
# Turn the middle cluster into a cycle. Cycle currently created using exact
# TSP solver
data = np.array([(terminal.x, terminal.y) for terminal in terminals])
kmeanAll.fit(data)

bestDist = float("inf")
centerCentroid = None
centralIndex = 0
for i in range(len(kmeanAll.cluster_centers_)):
    totalDist = 0
    
    for neighbor in kmeanAll.cluster_centers_:
        totalDist += distance(kmeanAll.cluster_centers_[i], neighbor)
    
    if totalDist < bestDist:
        bestDist = totalDist
        centerCentroid = kmeanAll.cluster_centers_[i]
        centralIndex = i

# Create a cycle using the nodes in the middle cluster
cycleNodes = []

for i in range(len(kmeanAll.labels_)):
    if kmeanAll.labels_[i] == centralIndex:
        cycleNodes.append(terminals[i])

distanceMatrix = []
for tail in cycleNodes:
    distances = []
    for head in cycleNodes:
        distances.append(distance(tail.getCoords(), head.getCoords()))
    distanceMatrix.append(distances)

cycle, _ = solve_tsp_dynamic_programming(np.array(distanceMatrix))
cycle.append(cycle[0])

# Cluster the remaining nodes separately
for node in cycleNodes:
    if node in A:
        A.remove(node)
    else:
        B.remove(node)

# Cluster the sources and sinks separately
data = np.array([(terminal.x, terminal.y) for terminal in A])
kmeanSources.fit(data)
data = np.array([(terminal.x, terminal.y) for terminal in B])
kmeanSinks.fit(data)

# Create an MST between points of the same type in the same cluster
for i in range(CLUSTERS):
    ACluster = []
    BCluster = []
    for j in range(len(kmeanSources.labels_)):
        if kmeanSources.labels_[j] == i:
            ACluster.append(A[j])
    for j in range(len(kmeanSinks.labels_)):
        if kmeanSinks.labels_[j] == i:
            BCluster.append(B[j])

    edges_to_add = prims(ACluster)
    for edge in edges_to_add:
        edges[edge[0]].append(edge[1])

    edges_to_add = prims(BCluster)
    for edge in edges_to_add:
        edges[edge[0]].append(edge[1])

    # Connect the clusters with the middle cycle
    minEdge = findMinEdgeBetweenClusters(cycleNodes, ACluster)
    edges[minEdge[0]].append(minEdge[1])
    minEdge = findMinEdgeBetweenClusters(cycleNodes, BCluster)
    edges[minEdge[0]].append(minEdge[1])


# Calculate the total length of the network
print("Total cost of the network:", totalCost(edges))

upperBoundMST = prims(terminals)
upperBound = 0
for edge in upperBoundMST:
    upperBound += distance(edge[0].getCoords(), edge[1].getCoords())
print("Upper bound of minimum spanning tree between all terminals:", upperBound)

# Visualization
# Draw sources
terminals_x = np.array([terminal.x for terminal in A])
terminals_y = np.array([terminal.y for terminal in A])
plt.scatter(terminals_x, terminals_y, s=[20 for i in range(len(terminals_x))], color="blue")
# Draw sinks
terminals_x = np.array([terminal.x for terminal in B])
terminals_y = np.array([terminal.y for terminal in B])
plt.scatter(terminals_x, terminals_y, s=[20 for i in range(len(terminals_x))], color="red")
# Draw cycle nodes
terminals_x = np.array([terminal.x for terminal in cycleNodes])
terminals_y = np.array([terminal.y for terminal in cycleNodes])
plt.scatter(terminals_x, terminals_y, s=[10 for i in range(len(terminals_x))], color="gray")

# Draw lines
# other clusters
for tail in edges:
    for head in edges[tail]:
        if tail in A:
            x = (tail.x, head.x)
            y = (tail.y, head.y)
            plt.plot(x, y, color="blue")
        else:
            x = (tail.x, head.x)
            y = (tail.y, head.y)
            plt.plot(x, y, color="blue")

# centre cluster
for i in range(len(cycle) - 1):
    x = (cycleNodes[cycle[i]].x, cycleNodes[cycle[i+1]].x)
    y = (cycleNodes[cycle[i]].y, cycleNodes[cycle[i+1]].y)
    plt.plot(x, y, color="gray")

plt.show()