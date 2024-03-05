import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# Read in an image
img = cv.imread('task1\img.png')
cv.imshow('owl', img)

b,g,r=cv.split(img)
cv.imshow('blue',b)
blank=np.zeros(img.shape[:2],dtype=np.uint8)
blue=cv.merge([b,blank,blank])
cv.imshow('blue',blue)
# Median Blur
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)


blank = np.zeros(img.shape[:2], dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise AND', bitwise_and)


bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('Bitwise OR', bitwise_or)
masked = cv.bitwise_and(img,img,mask=circle)
cv.imshow(' Masked Image', masked)

cv.waitKey(0)
plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])

plt.show()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel 
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)
cv.waitKey(0)