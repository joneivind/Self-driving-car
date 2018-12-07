import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import math

class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
    '''
    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])
    '''
    def __str__(self):
        return str([x.id for x in self.adjacent])

    def get_neighbors(self, node):
        return str([node.id for node in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def get_neighbors(self, node):
        return str([node.id for node in self.adjacent])

    def get_cost(self, from_node, to_node):
        v = Vertex(from_node)
        v.get_weight(to_node)
        return self.weights.get(to_node, 1)

import heapq

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]


def get_neighbors(graph, current_node):

    current_neighbors = []

    for v in graph:
        c_node = None
        n_node = None
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            c_node = vid
            n_node = wid
            if c_node == current_node:
                current_neighbors.append(n_node)
    
    return current_neighbors

def get_cost(graph, current_node, next_node):

    cost = 0

    for v in graph:
        c_node = None
        n_node = None
        for w in v.get_connections():
            vid = v.get_id()
            wid = w.get_id()
            c_node = vid
            n_node = wid
            if c_node == current_node:
                if n_node == next_node:
                    cost = v.get_weight(w)

    return cost

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2) # Manhattan Distance
    #return math.sqrt( (b[0]-a[0])**2 + (b[1]-a[1])**2 ) # Euclidean Distance

def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break

        for i, next in enumerate(get_neighbors(graph, current)):
            #print(get_cost(graph, current, next))
            new_cost = cost_so_far[current] + get_cost(graph, current, next)

            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):

    current = goal
    path = []
    
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    
    return path



'''
### PARAMETERS ###
vor.regions
vor.max_bound
vor.ndim
vor.ridge_dict
vor.ridge_points
vor.ridge_vertices
vor.npoints
vor.point_region
vor.points
vor.vertices
'''


points = np.array([[-1, 0.5], [3, 3], [0.0, 2.5], [3.0, 0.0], [0.0, -1.0],
                   [1, 0], [1, 3],
                   [2, 0], [2, 3]])


vor = Voronoi(points)

# Draw infinity vertices
center = points.mean(axis=0)
for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
    simplex = np.asarray(simplex)
    if np.any(simplex < 0):
        i = simplex[simplex >= 0][0] # finite end Voronoi vertex
        t = points[pointidx[1]] - points[pointidx[0]]  # tangent
        t = t / np.linalg.norm(t)
        n = np.array([-t[1], t[0]]) # normal
        midpoint = points[pointidx].mean(axis=0)
        far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 1.2
        plt.plot([vor.vertices[i,0], far_point[0]], [vor.vertices[i,1], far_point[1]], '--', color=(0.8, 0.8, 0.8))

g = Graph()

g_nodes = []

# Draw edges
for i, vpair in enumerate(vor.ridge_vertices):
    if vpair[0] >= 0 and vpair[1] >= 0:
        v0 = vor.vertices[vpair[0]]
        v1 = vor.vertices[vpair[1]]

        if [v0[0], v0[1]] not in g_nodes:
            g_nodes.append([v0[0], v0[1]])

        if [v1[0], v1[1]] not in g_nodes:
            g_nodes.append([v1[0], v1[1]])

        #print("Point A: " + str(v0) + "\t\tPoint B: " + str(v1) + "\t\tDistance: "+ str(distance))

        # Draw a line from v0 to v1.
        plt.plot([v0[0], v1[0]], [v0[1], v1[1]], color=(0, 0, 0), linewidth=2)

# Draw points
plt.plot(vor.points[:,0], vor.points[:, 1], 'o', ms=8, label='Road markers')

# Mark the Voronoi vertices.
plt.plot(vor.vertices[:,0], vor.vertices[:, 1], 'ro', ms=8, label='Voronoi vertices')

# Add nodes/vertecies
for n in g_nodes:
    g.add_vertex(str(n))

# Add edges to graph
for i, vpair in enumerate(vor.ridge_vertices):
    if vpair[0] >= 0 and vpair[1] >= 0:
        v0 = vor.vertices[vpair[0]]
        v1 = vor.vertices[vpair[1]]

        # Euclidean distance
        e_distance = math.sqrt( (v1[0]-v0[0])**2 + (v1[1]-v0[1])**2 )

        # Manhattan distance
        dx = v1[0] - v0[0]
        dy = v1[1] - v0[1]
        m_distance = abs(dx) + abs(dy)

        # Add egde to graph
        g.add_edge((v0[0], v0[1]), (v1[0], v1[1]), e_distance)


g_nodes = sorted(g_nodes)

start, goal = (g_nodes[0][0], g_nodes[0][1]), (g_nodes[-1:][0][0], g_nodes[-1:][0][1])
#start = (0,0)
#goal = (2.5, 2)

came_from, cost_so_far = a_star_search(g, start, goal)

path = reconstruct_path(came_from, start, goal)
print("Path: " + str(path))

for l in path:
    plt.plot(l[0], l[1], 'bx', ms=12)

# dummy for plotting label
plt.plot(l[0], l[1], 'bx', ms=12, label='Path')

plt.title('Voronoi diagram with A* path finder')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid(True)
plt.legend(loc='best')
plt.show()