import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def dist(p1, p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
def getAngle(p0, p1, p2):
	return math.acos((dist(p0, p1)**2 + dist(p0, p2)**2 - dist(p1, p2)**2)/(2*(dist(p0, p1)**2)*(dist(p0, p2)**2)))

def getNormalisedCoordinates(contour):
	# Finding the coordinates of the contours and their area
	coordinates = {}
	for i in range(len(contours)):
		cnt = contours[i]
		M = cv2.moments(cnt)
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

	return x, y

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

def getRedChannel(img):
	# image_copy_blue = img.copy()
	# # Setting Green and Red channel 0
	# image_copy_blue[:, :, 1] = 0
	# image_copy_blue[:, :, 2] = 0

	# Setting Green and Blue channel 0
	image_copy = img.copy()
	image_copy[:, :, 1] = 0
	image_copy[:, :, 0] = 0
	return image_copy
	
def dodo():        
	# Reading the template 
	img = cv2.imread("Gemini.png")
	cv2.imshow('original' ,img)
	red_channel = getRedChannel(img)
	plotImage(red_channel, "red")
	thresh = binariseImage(red_channel, [165.75, 191.25])

	# Subtracting to get only stars
	final = thresh[0] - thresh[1]
	plotImage(final, "final")

	stars = applyMedian(final, 3)
	plotImage(stars, "stars")

	stars_grey = getGrayscale(stars)
	final_stars = binariseImage(stars_grey, [20])
	final_stars_inverted = invertImage(final_stars[0])
	plotImage(final_stars[0], "final stars")

	edged = findEdges(final_stars_inverted, 30, 200)
	plotImage(edged, "edges")
	cv2.waitKey(0)  

	# Finding the contours in the image
	edge_copy = edged.copy()
	contours, hierarchy = cv2.findContours(edge_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

	print("Number of Contours found = " + str(len(contours)))
	# cv2.drawContours(img, contours, -1, (0, 255, 0), 3) 
	# cv2.imshow('Contours', img) 
	# cv2.waitKey(0) 
	# cv2.destroyAllWindows()

	x, y = getNormalisedCoordinates(contours)

	plt.figure("Normalised stars")
	plt.scatter(x, y)
	plt.show()


if __name__ == "__main__":
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

	print("Number of Contours found = " + str(len(contours)))

	# x, y = getNormalisedCoordinates(contours)

	# plt.figure("Normalised stars")
	# plt.scatter(x, y)
	# plt.show()

	cv2.waitKey()
	cv2.destroyAllWindows()
