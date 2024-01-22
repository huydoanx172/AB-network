import heapq
from kmeans import distance

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

def cycle(points):
    # # Return the sequence of terminals to form a cycle from the given terminals
    # # Currently using a greedy algorithm
    # distances = {}
    # visited = []
    # for i in range(len(points)):
    #     distances[points[i]] = []
    #     for j in range(len(points)):
    #         if i == j:
    #             distances[points[i]].append((points[i], float("inf")))
    #         else:
    #             distances[points[i]].append((points[j], distance(points[i].getCoords(), points[j].getCoords())))

    # for i in range(len(points)):
        
    # return visited
    pass