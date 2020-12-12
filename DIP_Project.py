import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# Reading the template 
img = cv2.imread("Gemini.png")
cv2.imshow('original' ,img)

image_copy_blue = img.copy()
# Setting Green and Red channel 0
image_copy_blue[:, :, 1] = 0
image_copy_blue[:, :, 2] = 0

# Setting Green and Blue channel 0
image_copy = img.copy()
image_copy[:, :, 1] = 0
image_copy[:, :, 0] = 0

# Thresholding the Image to binarise it
ret,thresh1 = cv2.threshold(image_copy,165.75,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(image_copy,191.25,255,cv2.THRESH_BINARY)

# Subtracting to get only stars
final = thresh1 - thresh2
# cv2.imshow('final', final)
# plt.imshow(final)
# plt.show()

# Applying median filter to get rid of noise left from words in the image
stars = cv2.medianBlur(final, 3)
# cv2.imshow('median', stars)
# plt.imshow(stars)
# plt.show()

# Converting it back to black and white image
stars_grey = cv2.cvtColor(stars, cv2.COLOR_BGR2GRAY)
ret,final_stars = cv2.threshold(stars_grey,20,255,cv2.THRESH_BINARY)
final_stars_inverted = cv2.bitwise_not(final_stars)
cv2.imshow('final stars', final_stars)
plt.imshow(final_stars, cmap='gray')
plt.show()

# Finding edges using Canny edge detection
edged = cv2.Canny(final_stars, 30, 200)
cv2.imshow("After Canny", edged)
# cv2.waitKey(0)  

# Finding the contours in the image
edge_copy = edged.copy()
contours, hierarchy = cv2.findContours(edge_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

print("Number of Contours found = " + str(len(contours)))
# cv2.drawContours(img, contours, -1, (0, 255, 0), 3) 
# cv2.imshow('Contours', img) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

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

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
def getAngle(p0, p1, p2):
    return math.acos((dist(p0, p1)**2 + dist(p0, p2)**2 - dist(p1, p2)**2)/(2*(dist(p0, p1)**2)*(dist(p0, p2)**2)))

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

# Plotting the visualised descriptor of the constellation
plt.figure("Normalised stars")
plt.scatter(x, y)
plt.show()