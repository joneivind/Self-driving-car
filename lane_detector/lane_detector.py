#!/usr/bin/python
#
# Dependencys:
# - Numpy
# - Matplotlib
# - Opencv (Tested with v3.4.1)
# - Pillow
# - ffmpeg or gstreamer
#
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import PID
import random
import sys
from scipy.spatial import Voronoi
from scipy.interpolate import CubicSpline

# Init PID controller
pid = PID.PID(0.1, 0.01, 0.0)
pid.SetPoint = 0.0
pid.setSampleTime(0.01)

# Draw a point
def draw_point(img, p, color ) :
	cv2.circle(img,p, 5, color, -1)

# Check if a point is inside a rectangle
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

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :
 
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :
         
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
         
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
         
            cv2.line(img, pt1, pt2, delaunay_color, 1)
            cv2.line(img, pt2, pt3, delaunay_color, 1)
            cv2.line(img, pt3, pt1, delaunay_color, 1)

    return img

# Draw voronoi diagram
def draw_voronoi(img, subdiv) :
 
    ( facets, centers) = subdiv.getVoronoiFacetList([])
 
    for i in xrange(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
    	
    	#print(facets[i][1][0])
        #cv2.imshow('Frame2', img)
         
        ifacet = np.array(ifacet_arr, np.int)
        color = (0,0,0)
 
        cv2.fillConvexPoly(img, ifacet, color);
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (255, 255, 0), 3)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), 1)

        #for j in xrange(0,len(facets[i])) :
        #    cv2.circle(img, (facets[i][j][0], facets[i][j][1]), 5, (0, 0, 255), -1)	

    return img
 
def region_of_interest(img, vertices):
	mask = np.zeros_like(img)
	match_mask_color = 255
	
	cv2.fillPoly(mask, vertices, match_mask_color)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def draw_lines(img, lines, color=[0, 255, 0], thickness=3):
    # If there are no lines to draw, exit.
        if lines is None:
            return
    # Make a copy of the original image.
	img = np.copy(img)
    # Create a blank image that matches the original in size.
	line_img = np.zeros(
		(
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
	img = cv2.addWeighted(img, 1, line_img, 1.0, 0.0)
    # Return the modified image.
	return img

def pipeline(image):

	# Get the hight and width of the image
	height = image.shape[0]
	width = image.shape[1]
	
	# Create a region of interest
	region_of_interest_vertices = [
	    (0, height),
	    (width / 2, height / 2),
	    (width, height),
	]

	# Convert to grayscale here.
	gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# Call Canny Edge Detection here.
	cannyed_image = cv2.Canny(gray_image, 100, 200)

	# Crop image to region of interest
	cropped_image = region_of_interest(
	    cannyed_image,
	    np.array([region_of_interest_vertices], np.int32),
	)

	# Find coordinates to all lines in image with Hough lines, format[x1, y1, x2, y2]
	lines = cv2.HoughLinesP(
	    cropped_image,
	    rho=6,
	    theta=np.pi / 60,
	    threshold=160,
	    lines=np.array([]),
	    minLineLength=50,
	    maxLineGap=80
	)

	# Separate line placeholders
	left_line_x = []
	left_line_y = []
	right_line_x = []
	right_line_y = []

	# Group lines
	for line in lines:
		for x1, y1, x2, y2 in line:

			slope = float(y2 - y1) / float(x2 - x1) # Calculating the slope.
			if math.fabs(slope) < 0.5: # Only consider extreme slope
				continue

			if slope <= 0: # If the slope is negative, left group.
				left_line_x.extend([x1, x2])
				left_line_y.extend([y1, y2])

			else: # Otherwise, right group.
				right_line_x.extend([x1, x2])
				right_line_y.extend([y1, y2])

	min_y = image.shape[0]/2 + image.shape[0]/8 # Just below the horizon
	max_y = image.shape[0] # The bottom of the image

	poly_left = np.poly1d(np.polyfit(
		left_line_y,
		left_line_x,
		deg=1
	))

	left_x_start = int(poly_left(max_y))
	left_x_end = int(poly_left(min_y))

	poly_right = np.poly1d(np.polyfit(
		right_line_y,
		right_line_x,
		deg=1
	))

	right_x_start = int(poly_right(max_y))
	right_x_end = int(poly_right(min_y))


	#############################
	### Voronoi overlay start ###
	#############################

	# Create an array of points.
	points = []

	# Rectangle to be used with Subdiv2D
	rect = (0, 0, width+10, height+10)

	# Create an instance of Subdiv2D
	subdiv = cv2.Subdiv2D(rect);

	# Add some point along the lanes
	for point in range(12) :
		points.append((left_x_start + int(0.1*point*(left_x_end-left_x_start)), (max_y) + int(0.1*point*(min_y-max_y))))
		points.append((right_x_start + int(0.1*point*(right_x_end-right_x_start)), (max_y) + int(0.1*point*(min_y-max_y))))
		#font = cv2.FONT_HERSHEY_SIMPLEX
		#cv2.putText(image, str(point),((left_x_start + int(0.2*point*(left_x_end-left_x_start))-40, (max_y) + int(0.2*point*(min_y-max_y))+5)), font, 0.7,(255,255,255),1,cv2.LINE_AA)
		#cv2.putText(image, str(point),((right_x_start + int(0.2*point*(right_x_end-right_x_start)+25), (max_y) + int(0.2*point*(min_y-max_y))+5)), font, 0.7,(255,255,255),1,cv2.LINE_AA)

	
	# Obstacle points
	#points.append((left_x_end + (left_x_start-left_x_end)/5 , (min_y) + (max_y-min_y)/2))
	#points.append((left_x_end + (left_x_start-left_x_end)/4 + 50, (min_y) + (max_y-min_y)/4))
	#points.append((left_x_end + (left_x_start-left_x_end)/3 + 50, (min_y) + (max_y-min_y)/3))
	#points.append((left_x_end + (left_x_start-left_x_end)/2 + 50, (min_y) + (max_y-min_y)/2))
	






	### Create voronoi path ###

	rect2 = (left_x_start, left_x_end, right_x_end, right_x_start)

	vor = Voronoi(points)
	list_vertices = []

	# Remove points out of bound
	for vertice in vor.vertices:
		v_point = ((int(round(vertice[1])), int(round(vertice[0]))))
		if rect_contains(rect2, v_point):
		#if vertice[1] > 0 and vertice[1] < height and vertice[0] > 0 and vertice[0] < width:
			for simplex in vor.ridge_vertices:
				simplex = np.asarray(simplex)
				if np.all(simplex >= 0):
					list_vertices.append(v_point)

	# Sort valid vertices list
	list_vertices = sorted(list_vertices)

	# End position
	x = [min_y]
	y = [(left_x_end + (right_x_end-left_x_end)/2)]

	for item in list_vertices:
		if x.count(item[0]) == 0 and y.count(item[1]) == 0:
			x.append(item[0])
			y.append(item[1])

	# Start position
	x.append(height+50)
	y.append(width/2)

	
	cs = CubicSpline(x, y)
	cx = np.arange(x[0], x[-1], 1)
	cy = cs(cx)

	for p, _ in enumerate(cx):
		cv2.circle(image, (int(round(cy[p])), int(round(cx[p]))), 2, (255,255,255), -1)
	
	#for p in list_vertices:
	#	cv2.circle(image, (p[1], p[0]), 5, (255,255,255), -1)







	# Insert points into subdiv
	for p in points :
		subdiv.insert(p)

	
	# Allocate space for Voronoi Diagram
	img_voronoi = np.zeros(image.shape, dtype = image.dtype)

	#overlay_voronoi_diag = image.copy()
	opacity = 0.3

	# Draw Voronoi diagram onto image
	#voronoi_diag_img = draw_voronoi(img_voronoi,subdiv)
	#cv2.addWeighted(voronoi_diag_img, opacity, image, 1 - opacity, 0, image)
	
	# Draw delaunay triangles
	#img_dela = draw_delaunay( image, subdiv, (255, 255, 255) )
	#cv2.addWeighted(img_dela, opacity, image, 1 - opacity, 0, image)

	###########################
	### Voronoi overlay end ###
	###########################

	# Add color to the road
	overlay = image.copy()
	opacity = 0.1
	contours = np.array( [ [left_x_start, max_y], [left_x_end, min_y], [right_x_end, min_y], [right_x_start, max_y] ] )
	cv2.fillPoly(overlay, pts =[contours], color=(0,255,0))
	cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)


	# Draw lane lines
	image = draw_lines(
		image,
		[[
			[left_x_start, max_y, left_x_end, min_y],
			[right_x_start, max_y, right_x_end, min_y],
		]],
		thickness=5,
	)


	'''
	# Draw zones between points
	for point in range(6) :
		cv2.line(line_image,(left_x_start + int(0.2*point*(left_x_end-left_x_start)), (max_y) + int(0.2*point*(min_y-max_y))), (right_x_start + int(0.2*point*(right_x_end-right_x_start)), (max_y) + int(0.2*point*(min_y-max_y))),(0,255,0),1)
	'''
	# Draw the points on the road lanes
	#for p in points :
	#	draw_point(image, p, (0,255,0))


	# Calculate offset error
	center_point = ((right_x_end + (right_x_start-right_x_end)/2) - (left_x_end + (left_x_start-left_x_end)/2))/2 + (left_x_end + (left_x_start-left_x_end)/2)
	ref_point = width/2
	offset_error = ref_point - center_point

	'''
	# Show error point on image
	cv2.line(line_image,(center_point,height-(max_y-min_y)/2),(ref_point,height-(max_y-min_y)/2),(255,0,0),2)
	cv2.circle(line_image,(width/2, (min_y) + (max_y-min_y)/2), 8, (0,255,0), -1)
	cv2.circle(line_image,(ref_point - offset_error, (min_y) + (max_y-min_y)/2), 8, (255,0,0), -1)
	'''

	# Update PID
	pid_output = round(pid.update(offset_error),2)

    # Clear terminal and print
	#sys.stderr.write("\x1b[2J\x1b[H")
	#print('*** Path finder ***\n')
	#print('PID: ' + str(pid_output) + '\tError: ' + str(offset_error) + '\n')
	print("\n")
	# Print some data as text
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(image,"Error: " + str(-offset_error),(10,40), font, 0.8,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(image,"PID out: " + str(pid_output),(10,80), font, 0.8,(255,255,255),2,cv2.LINE_AA)


	return image



if __name__ == "__main__":
	
	'''
	# Get image file
	image = mpimg.imread('test4.jpg')
 
    # Show results    
	plt.figure()
	plt.imshow(pipeline(image))
	plt.show()
	'''

	# Get video file
	cap = cv2.VideoCapture('../samples/solidYellowLeft.mp4')

	# Check if camera opened successfully
	if (cap.isOpened()== False): 
	  print("Error opening video stream or file")
	
	# Read until video is completed
	while(cap.isOpened()):

		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
	 
	    	# Display the resulting frame
			cv2.imshow('Frame', pipeline(frame))
			#pipeline(frame)

	    	# Press Q on keyboard to  exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break

		# Break the loop
		else:
			break
	 
	# When everything done, release the video capture object
	cap.release()
	 
	# Closes all the frames
	cv2.destroyAllWindows()