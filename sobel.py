import cv2
import numpy as np

# Read an image
# Replace 'path_to_your_image.jpg' with the actual file path
image = cv2.imread('AIROGS_2024/images/TRAIN000000.JPG')
# Resize the image to 500x500
image = cv2.resize(image, (500, 500))
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculation of Sobelx
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)

# Calculation of Sobely
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)


# Combine Sobelx and Sobely
combined_edges = cv2.magnitude(sobelx, sobely)
# # Display Sobelx
# cv2.imshow('Sobelx', sobelx)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Display Sobely
cv2.imshow('Sobely', sobely)
cv2.waitKey(0)
cv2.destroyAllWindows()
