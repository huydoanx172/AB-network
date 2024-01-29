import heapq
import numpy as np

class PriorityQueue:
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        self.count -= 1
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def push(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, simply push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            entry = (priority, self.count, item)
            heapq.heappush(self.heap, entry)
            self.count += 1

def findMinCycle(points):
    # Return the sequence of terminals to form a cycle from the given terminals
    # Currently using a greedy algorithm
    distances = getEuclideanDistances(points)
    visited = []

    # Greedy algorithm starting from the first point
    visited.append(points[0])
    currPoint = points[0]
    while len(visited) != len(points):
        # remove the visited points from consideration
        toRemove = []
        for edge in distances[currPoint]:
            if edge[0] in visited:
                toRemove.append(edge)
        for edge in toRemove:
            distances[currPoint].remove(edge)
        
        minEdge = min(distances[currPoint], key=lambda x: x[1])
        visited.append(minEdge[0])
        currPoint = minEdge[0]

    visited.append(points[0])
    return visited

def getEuclideanDistances(points):
    """
    Given a list of Terminal objects, return a dictionary containing the 
    distances of all edges between terminals
    """
    distances = {}
    for i in range(len(points)):
        distances[points[i]] = []
        for j in range(len(points)):
            if i == j:
                distances[points[i]].append((points[i], float("inf")))
            else:
                distances[points[i]].append((points[j], distance(points[i].getCoords(), points[j].getCoords())))
    
    return distances

def distance(a, b):
    """
    Given two points a and b represented as tuples in the Euclidean plane, return
    the distance between them
    """
    dimensions = len(a)
    
    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return np.sqrt(_sum)

def findMinEdgeBetweenClusters(a, b):
    """
    Given two set of terminals a and b, return the shortest edge between a
    point from a and a point from b
    """
    minDist = float("inf")
    minEdge = None

    for tail in a:
        for head in b:
            dist = distance(tail.getCoords(), head.getCoords())
            if dist < minDist:
                minDist = dist
                minEdge = (tail, head)

    if not minEdge:
        print("error in findMinEdge function")
        print("i cluster:", a)
        print("j cluster:", b)
    return minEdge

def totalCost(edges):
    """
    Given a dictionary of edges in a graph in the Euclidean plane, return the
    total cost of all the edges
    """
    totalDist = 0
    for tail in edges:
        for head in edges[tail]:
            totalDist += distance(tail.getCoords(), head.getCoords())
    return totalDist

def cluster(points, num_clusters=6, type="kmeans"):
    """
    Given a list of terminals, cluster them based on the given clustering
    algorithm and number of clusters. Then, return a dictionary with keys
    corresponding to the cluster indexes and the value associated is the list
    of indexes of points belonging to that cluster
    """

    if type == "kmeans":
        pass