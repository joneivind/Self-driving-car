class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def path_planner(start=(0,2), end=(5,0), obstacle_coords=[]) :

    # Road split into sections
    road_map = [    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]

    # Road sections as rectangles [x0,y0,x1,y1]
    sections =  [   [[0, 0, 10, 10],  [10, 0, 20, 10],  [20, 0, 30, 10]],
                    [[0, 10, 10, 20], [10, 10, 20, 20], [20, 10, 30, 20]],
                    [[0, 20, 10, 30], [10, 20, 20, 30], [20, 20, 30, 30]],
                    [[0, 30, 10, 40], [10, 30, 20, 40], [20, 30, 30, 40]],
                    [[0, 40, 10, 50], [10, 40, 20, 50], [20, 40, 30, 50]],
                    [[0, 50, 10, 60], [10, 50, 20, 60], [20, 50, 30, 60]],
                ]

    # Ostacles coordinates as points (x,y)
    obstacle_coords = [ (0,5), 
                        (1,1), 
                        (25,33),
                        (9,45),
                        (15,15),
                    ]

    # Check for obstacles in each road section and mark on road map
    for obstacle in obstacle_coords:
        for x, rectangles in enumerate(sections):
            for y, rectangle in enumerate(rectangles):
                if rect_contains(rectangle, obstacle):
                    road_map[x][y] = 1

    # Find path from start to end
    path = astar(road_map, start, end)

    # Mark path on road map
    for i in path:
        if road_map[i[0]][i[1]] is not 1:
            road_map[i[0]][i[1]] = 2
    
    
    # Print some test data
    print('\nStart_section: '+str(start)+'\nEnd_section: '+str(end))
    print('\nObstacle_coordinates: \n'+str(obstacle_coords))
    print('\nPath: \n'+ str(path))
    print('\nRectangle_grid: (1=contain_obstacle, 2=path)')    
    for section in road_map:
        print(section)    

    # Plot path
    import numpy as np
    from scipy.interpolate import CubicSpline
    import matplotlib.pyplot as plt

    x = []
    y = []
    for i,j in path:
        x.append(i)
        y.append(j)

    xo = []
    yo = []
    for k, sections in enumerate(road_map):
        for l, section in enumerate(sections):
            if road_map[k][l] == 1:
                xo.append(k)
                yo.append(l)
    print('\n')
    cs = CubicSpline(x, y)
    xs = np.arange(-0.0, 5.1, 0.1)
    plt.figure(figsize=(6.5, 3))
    plt.plot(x, y, 'o', label='Waypoint')
    plt.plot(xs, cs(xs), label="Path")
    plt.plot(xo, yo, 'o', label='Obstacle')
    plt.xlim(-0.5, 5.5)
    plt.ylim(-1, 4)
    plt.legend(loc='upper left', ncol=2)
    plt.grid(True)
    plt.title('A* path planner')
    plt.xlabel('Road')
    plt.ylabel('Lanes')
    plt.show()    

    return path

if __name__ == '__main__':
    path_planner()