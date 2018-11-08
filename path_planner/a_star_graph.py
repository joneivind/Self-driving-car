from implementation import *

class GridWithWeights(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}
    
    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)


start, goal = (1, 4), (7, 8)
came_from, cost_so_far = a_star_search(diagram4, start, goal)