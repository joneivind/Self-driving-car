#!/usr/bin/python

from scipy.spatial import Voronoi
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

def main():

	points = np.array([	[0, -0.5], [0.6, -0.2], [1.2, 0], [1.8, 0], [2.4, -0.3], 
						[0.3, 1.8], [0.9, 1.9], [1.5, 2], [2.1, 1.9], [2.7, 1.8]])

	# Split into left and right lane
	lane_left = []
	lane_right = []
	center_line = 1.0
	
	for point in points:
		if point[1] >= center_line:
			lane_left.append((point[0], point[1]))
		else:
			lane_right.append((point[0], point[1]))


	# Interpolate left lane
	lane_left = sorted(lane_left)

	x_lane_left = [i[0] for i in lane_left]
	y_lane_left = [i[1] for i in lane_left]

	cs = CubicSpline(x_lane_left, y_lane_left)
	cx = np.arange(x_lane_left[0], x_lane_left[-1], 0.01)
	cy = cs(cx)

	plt.plot(cx, cy, "-", label="lane_left")
	plt.plot([i[0] for i in lane_left], [i[1] for i in lane_left], 'og')


	# Interpolate right lane
	lane_right = sorted(lane_right)

	x_lane_right = [i[0] for i in lane_right]
	y_lane_right = [i[1] for i in lane_right]	

	cs = CubicSpline(x_lane_right, y_lane_right)
	cx = np.arange(x_lane_right[0], x_lane_right[-1], 0.01)
	cy = cs(cx)

	plt.plot(cx, cy, "-", label="lane_right")
	plt.plot([i[0] for i in lane_right], [i[1] for i in lane_right], 'og')



	### Create voronoi path ###

	vor = Voronoi(points)
	list_vertices = []

	# Remove points out of bound
	for vertice in vor.vertices:
		if vertice[1] <= min(y_lane_left) and vertice[1] >= max(y_lane_right) and vertice[0] >= 0:
			plt.plot(vertice[0], vertice[1], 'or')
			list_vertices.append((vertice[0], vertice[1]))

	# Sort valid vertices list
	list_vertices = sorted(list_vertices)

	# Start position
	x = []
	y = []

	for item in list_vertices:
		x.append(item[0])
		y.append(item[1])

	# End position
	#x.append(2)
	#y.append(0.5)
		
	cs = CubicSpline(x, y)
	cx = np.arange(x[0], x[-1], .01)
	cy = cs(cx)

	#plt.plot(cx, cy, "-", label="path")

	
	# Remove dead ends
	for simplex in vor.ridge_vertices:
		simplex = np.asarray(simplex)
		if np.all(simplex >= 0):
			plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')
	
	
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
			far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
			plt.plot([vor.vertices[i,0], far_point[0]], [vor.vertices[i,1], far_point[1]], 'k--')
	
	
	plt.xlim(-1, 3); plt.ylim(-1, 3)
	plt.show()

if __name__ == "__main__":
	main()