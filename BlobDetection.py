import matplotlib.pyplot as plt
import numpy as np
import cv2

""" Program by: Ethan S
Date: 7/12/2022
Notes: Image detection program that finds particles in an image through Simple Blob Detection. Very simplistic method.
"""
image = cv2.imread("C:/Users/admin/Downloads/blob1.jpg")

params = cv2.SimpleBlobDetector_Params()

# Threshold
params.minThreshold = 0
params.maxThreshold = 200
# Area
params.filterByArea = True
params.minArea = 11  # by pixels
params.maxArea = 395
# Color
params.filterByColor = True
params.blobColor = 255  # 0 = black color, 255 = light
# Circularity
params.filterByCircularity = True
params.minCircularity = 0.2  # values between 0 and 1
params.maxCircularity = 0.6
# Inertia
params.filterByInertia = True
params.minInertiaRatio = 0  # values between 0 and 1
params.maxInertiaRatio = 1
# Convexity
params.filterByConvexity = True
params.minConvexity = .6  # values between 0 and 1
params.maxConvexity = .9
# Min Distance Between Blobs
params.minDistBetweenBlobs = 0

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(image)

# resize image
scale_percent = 30  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

img_with_blobs = cv2.drawKeypoints(image, keypoints, np.array([]),
                                   (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
resized = cv2.resize(img_with_blobs, dim, interpolation=cv2.INTER_AREA)
plt.imshow(img_with_blobs)
cv2.imshow("Blob Detection", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()