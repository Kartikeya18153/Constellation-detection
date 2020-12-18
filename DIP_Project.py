import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import pickle

def dist(p1, p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
def getAngle(p0, p1, p2):
	return math.acos((dist(p0, p1)**2 + dist(p0, p2)**2 - dist(p1, p2)**2)/(2*(dist(p0, p1)**2)*(dist(p0, p2)**2)))

def getNormalisedCoordinates(contours, lines=None):

	lines = np.array(lines)

	# Finding the coordinates of the contours and their area
	coordinates = {}
	for i in range(len(contours)):
		cnt = contours[i]
		M = cv2.moments(cnt)
		# print(M)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		area = cv2.contourArea(cnt)
		if area in coordinates:
			coordinates[area+0.0001] = (cx, cy)
		else:
			coordinates[area] = (cx, cy)
		
	sorted_area = []
	for i in reversed(sorted(coordinates.keys())):  
		sorted_area.append(i)

	# Creating coordinates for each star in the costellation
	x = []
	y = []
	for area in sorted_area:
		x.append(coordinates[area][0])
		y.append(coordinates[area][1])
	
	for line in lines:
		for x1,y1,x2,y2 in line:
			x1-= x[0]
			y1 -= y[0]
			x2 -= x[0]
			y2 -= y[0]
			line[0][0] = x1
			line[0][1] = y1
			line[0][2] = x2
			line[0][3] = y2

	# Shifting the brightest star to origin
	for i in range(len(x)):
		x[len(x)-i-1] -= x[0]
		y[len(x)-i-1] -= y[0]


	# Distance between brightest and second brightest star
	distance_brightest = math.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)

	# Normalising the distance between all stars
	for i in range(len(x)):
		x[i] = x[i]/distance_brightest
		y[i] = y[i]/distance_brightest
	lines = lines/distance_brightest

	# Finding the angle of rotation to rotate second brightest star to (1, 0)
	p0 = (x[0], y[0])
	p1 = (x[1], y[1])
	p2 = (1, 0)
	theta = getAngle(p0, p1, p2)
	if round(x[1]*math.cos(theta) - y[1]*math.sin(theta), 2) != float(1) or round(x[1]*math.sin(theta) + y[1]*math.cos(theta), 2) != float(0):
		theta = -theta

	# Updating the new coordinates of each star
	for i in range(1, len(x)):
		x_new = x[i]*math.cos(theta) - y[i]*math.sin(theta)
		y_new = x[i]*math.sin(theta) + y[i]*math.cos(theta)
		x[i], y[i] = round(x_new, 2), round(y_new, 2)

	for line in lines:
		for x1,y1,x2,y2 in line:
			x1_new = x1*math.cos(theta) - y1*math.sin(theta)
			y1_new = x1*math.sin(theta) + y1*math.cos(theta)
			x2_new = x2*math.cos(theta) - y2*math.sin(theta)
			y2_new = x2*math.sin(theta) + y2*math.cos(theta)
			line[0][0] = x1_new
			line[0][1] = y1_new
			line[0][2] = x2_new
			line[0][3] = y2_new
			
	return np.array(x), np.array(y)

# Finding edges using Canny edge detection
def findEdges(image, thresh1, thresh2):
	out = cv2.Canny(np.array(image), thresh1, thresh2)
	return out

def invertImage(image):
	return cv2.bitwise_not(image)

# Converting it back to black and white image
def getGrayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying median filter to get rid of noise left from words in the image
def applyMedian(image, size):
	return cv2.medianBlur(image, size)

# Plot the specified image with the name given
def plotImage(image, imagename="figure"):
	cv2.imshow(imagename, image)

def binariseImage(img, thresholds):
	# Thresholding the Image to binarise it
	output_thresh = []
	for threshold in thresholds:
		ret,thresh = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
		output_thresh.append(thresh)

	return output_thresh

def getGreenChannel(img):

	image_copy_green = img.copy()
	# Setting Green and Red channel 0
	image_copy_green[:, :, 0] = 0
	image_copy_green[:, :, 2] = 0
	return image_copy_green

def getBlueChannel(img):
	
	image_copy_blue = img.copy()
	# Setting Green and Red channel 0
	image_copy_blue[:, :, 1] = 0
	image_copy_blue[:, :, 2] = 0
	return image_copy_blue

def getRedChannel(img):

	# Setting Green and Blue channel 0
	image_copy = img.copy()
	image_copy[:, :, 1] = 0
	image_copy[:, :, 0] = 0
	return image_copy
	
def makeTemplates():        

	# Directory where the templates are stoed
	template_directory = "./Templates"
	templates_coordinates = {}

	# Iterate through each file in the template directory to process one template at a time
	for filename in os.listdir(template_directory):
	# for filename in ["Hercules.png"]:
		print(filename)

		# Reading the template 
		img = cv2.imread("./Templates/" + filename)
		# cv2.imshow('original' ,img)
		red_channel = getRedChannel(img)
		# plotImage(red_channel, "red")
		thresh = binariseImage(red_channel, [165.75, 191.25])

		blue_channel = getBlueChannel(img)

		thresh2 = binariseImage(red_channel, [125.75, 255])
		letters = thresh2[0] - thresh2[1]
		new_blue = blue_channel + letters

		for i in range(len(new_blue)):
			for j in range(len(new_blue[i])):
				if new_blue[i][j][2] != 0:
					new_blue[i][j][1] = 0
					new_blue[i][j][0] = 0
					new_blue[i][j][2] = 0

		# Subtracting to get only stars
		final = thresh[0] - thresh[1]
		# plotImage(final, "final")

		stars = applyMedian(final, 3)
		lines = applyMedian(new_blue, 3)

		stars_grey = getGrayscale(stars)
		lines_grey = getGrayscale(lines)
		final_stars = binariseImage(stars_grey, [20])
		final_lines = binariseImage(lines_grey, [5])
		final_stars_inverted = invertImage(final_stars[0])
		final_lines_inverted = invertImage(final_lines[0])
		final_lines = applyMedian(final_lines[0], 3)  

		edged = findEdges(final_stars_inverted, 30, 200)
		# plotImage(edged, "edges")
		# cv2.waitKey(0)  

		rho = 1  # distance resolution in pixels of the Hough grid
		theta = np.pi / 180  # angular resolution in radians of the Hough grid
		threshold = 10  # minimum number of votes (intersections in Hough grid cell)
		min_line_length = 2  # minimum number of pixels making up a line
		max_line_gap = 3  # maximum gap in pixels between connectable line segments
		line_image = np.copy(img) * 0  # creating a blank to draw lines on

		# Run Hough on edge detected image
		# Output "lines" is an array containing endpoints of detected line segments
		drawn_lines = cv2.HoughLinesP(final_lines, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
		# drawn_lines = drawn_lines[1:]
		for line in drawn_lines:
			for x1,y1,x2,y2 in line:
				cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
		# cv2.imshow("drawn", line_image)
		# cv2.waitKey(0)

		# Finding the contours in the image
		edge_copy = edged.copy()
		contours, hierarchy = cv2.findContours(edge_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

		final_contours = []
		for contour in contours:
			area = cv2.contourArea(contour)
			if area != 0:
				final_contours.append(contour)
		
		print("Number of Contours found = " + str(len(final_contours)))
		# cv2.drawContours(img, contours, -1, (0, 255, 0), 3) 
		# cv2.imshow('Contours', img) 
		# cv2.waitKey(0) 
		# cv2.destroyAllWindows()

		x, y, normalised_lines = getNormalisedCoordinates(final_contours, drawn_lines)

		templates_coordinates[filename[:-4]] = (x, y)

		# Plot the normalised stars or save them
		plt.figure("Normalised " + filename[:-4] + " stars")
		plt.scatter(x, y)
		for line in normalised_lines:
			for x1,y1,x2,y2 in line:
				plt.plot([x1, x2], [y1, y2], color='red')

		# plt.savefig("./Normalised_Templates/" + filename)
		# plt.close()
		plt.show()
		# exit()

		# Return the normalised coordinates 
		# return x, y
	
	# Save the normalised coordinates for all templates
	with open("Template Coordinates", "wb") as fp:
		pickle.dump(templates_coordinates, fp)

def test():

	# Process and find the normalised coordinate for each template present in the Templates directory
	makeTemplates()

	img = cv2.imread("test.png")
	img = getGrayscale(img)
	cv2.imshow('test_img' ,img)

	a = 50
	thresh = binariseImage(img, [190])
	# Subtracting to get only stars
	final = thresh[0]
	plotImage(final, "final")

	stars = applyMedian(final, 5)
	# plotImage(stars, "stars")

	# stars_grey = getGrayscale(stars)
	# final_stars = binariseImage(stars, [70])
	# final_stars_inverted = invertImage(final_stars[0])
	final_stars_inverted = invertImage(stars)
	# plotImage(final_stars[0], "final stars")

	edged = findEdges(final_stars_inverted, 30, 200)
	plotImage(edged, "edges")

	edge_copy = edged.copy()
	contours, hierarchy = cv2.findContours(edge_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

	final_contours = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if area != 0:
			final_contours.append(contour)

	print("Number of Contours found = " + str(len(final_contours)))

	x, y = getNormalisedCoordinates(final_contours)

	plt.figure("Normalised stars")
	plt.scatter(x, y)
	plt.show()

	return x , y

def score(x , y , template_x , template_y) :
	test_coordinates = {}
	template_coordinates = {}
	
	for i in range(len(x)):
		test_coordinates[x[i]] = y[i]

	for i in range(len(template_x)):
		template_coordinates[template_x[i]] = template_y[i]

	sorted_x = []
	sorted_y = []
	sorted_template_x = []
	sorted_template_y = []

	for i in sorted(test_coordinates.keys()):
		sorted_x.append(i)
		sorted_y.append(test_coordinates[i])
	
	for i in sorted(template_coordinates.keys()):
		sorted_template_x.append(i)
		sorted_template_y.append(template_coordinates[i])
	
	print(sorted_template_x)
	print(sorted_template_y)
	print("test")
	print(sorted_x)
	print(sorted_y)
	
	index_test = 0
	index_template = 0

	count = 0
	matched_coord = []

	while index_test < len(sorted_x):
		check_x = sorted_template_x[index_template]
		check_y = sorted_template_y[index_template]
		
		if sorted_x[index_test] <= check_x + 0.05 and sorted_x[index_test] >= check_x - 0.05 and sorted_y[index_test] <= check_y + 0.05 and sorted_y[index_test] >= check_y - 0.05:
			count += 1
			index_template += 1
			matched_coord.append((sorted_x[index_test], sorted_y[index_test]))
		
		elif sorted_x[index_test] > check_x:
			index_template += 1

		index_test += 1
	print(count)
	print(matched_coord)

	cv2.waitKey()
	cv2.destroyAllWindows()

def simillarity_error(train ,test):
	threshold = 0.05
	error = 0
	count = 0
	for i in range(train[0].shape[0]) :
		distances = np.sqrt((test[0] - train[0][i]) **2 + (test[1] - train[1][i]) **2)
		min_dist = min(distances)
		if(min_dist < threshold) :
			count += 1
			error += min_dist

	return count , error


if __name__ == "__main__":
	x_test , y_test = test()

	file = open('Template Coordinates' , 'rb')
	template_coordinate = pickle.load(file)

	x_template , y_template = template_coordinate['Gemini']

	# x_template , y_template = np.array(x_template) , np.array(y_template)

	e = simillarity_error((x_template, y_template) , (x_test , y_test))
	score(x_test , y_test , x_template , y_template)
	print(e)