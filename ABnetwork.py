from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from util import PriorityQueue, getEuclideanDistances, distance, findMinEdgeBetweenClusters, totalCost
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing
import sys

NUM_POINTS = 50
# CLUSTERS = 7
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

upperBounds = []
for CLUSTERS in range(4, 21):
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
    kmeanAll = KMeans(n_clusters=CLUSTERS, n_init=10)

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
    for i in range(len(cycle) - 1):
        edges[cycleNodes[cycle[i]]].append(cycleNodes[cycle[i+1]])
        cycleNodes[cycle[i]].deg += 1
        cycleNodes[cycle[i+1]].deg += 1

    # Create an MST between points in the same cluster and connect the tree with the
    # middle cycle
    minEdges = []
    for i in range(CLUSTERS):
        if i == centralIndex:
            continue

        clusterNodes = []
        for j in range(len(kmeanAll.labels_)):
            if kmeanAll.labels_[j] == i:
                clusterNodes.append(terminals[j])

        edgesToAdd = prims(clusterNodes)
        for edge in edgesToAdd:
            edges[edge[0]].append(edge[1])
            edge[0].deg += 1
            edge[1].deg += 1
        minEdge = findMinEdgeBetweenClusters(cycleNodes, clusterNodes)
        # Include i so it's easier to retrieve connected cluster later
        minEdges.append((minEdge, i))

    # Find the cluster that is closest to the middle cycle first, because at least
    # one cluster has to connect to the middle cycle
    minMinEdge = min(minEdges, key=lambda x: distance(x[0][0].getCoords(), x[0][1].getCoords()))
    edges[minMinEdge[0][0]].append(minMinEdge[0][1])
    connectedIndex = minMinEdge[1]

    # For the other clusters, either connect to the middle cycle or the nearest cluster
    for i in range(CLUSTERS):
        if i == centralIndex or i == connectedIndex:
            continue

        ithClusterNodes = []
        for j in range(len(kmeanAll.labels_)):
            if kmeanAll.labels_[j] == i:
                ithClusterNodes.append(terminals[j])
        minClusterEdges = []
        for j in range(CLUSTERS):
            if j in (centralIndex, connectedIndex, i):
                continue
            
            jthClusterNodes = []
            for k in range(len(kmeanAll.labels_)):
                if kmeanAll.labels_[k] == j:
                    jthClusterNodes.append(terminals[k])

            minClusterEdge = findMinEdgeBetweenClusters(ithClusterNodes, jthClusterNodes)
            minClusterEdges.append(minClusterEdge)
        
        minMinClusterEdge = min(minClusterEdges, key=lambda x: distance(x[0].getCoords(), x[1].getCoords()))
        cycleEdge = findMinEdgeBetweenClusters(ithClusterNodes, cycleNodes)
        edgeToAdd = min((minMinClusterEdge, cycleEdge), key=lambda x: distance(x[0].getCoords(), x[1].getCoords()))
        # print(f"({edgeToAdd[0]}, {edgeToAdd[1]})")
        edges[edgeToAdd[0]].append(edgeToAdd[1])
            

    # Calculate the total length of the network
    print(f"Total cost of the network using kmeans with {CLUSTERS} clusters:", totalCost(edges))

    # Calculate the upper bound which is a TSP of all the terminals
    distanceMatrix = []
    for tail in terminals:
        distances = []
        for head in terminals:
            distances.append(distance(tail.getCoords(), head.getCoords()))
        distanceMatrix.append(distances)

    upperBoundTSP, upperBound = solve_tsp_simulated_annealing(np.array(distanceMatrix))
    upperBounds.append(upperBound)

print("Upper bound of approximate TSP between all terminals:", min(upperBounds))

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
        x = (tail.x, head.x)
        y = (tail.y, head.y)
        plt.plot(x, y, color="orange")

# centre cluster
for i in range(len(cycle) - 1):
    x = (cycleNodes[cycle[i]].x, cycleNodes[cycle[i+1]].x)
    y = (cycleNodes[cycle[i]].y, cycleNodes[cycle[i+1]].y)
    plt.plot(x, y, color="gray")

plt.show()