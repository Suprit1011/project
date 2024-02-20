
import cv2
import numpy as np

def main():
    # Load the image
    image = cv2.imread(r'tasktwo\lineimg.png')
    cv2.imshow('original',image) 
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    dilated = cv2.dilate(edges, (5,5), iterations=3)
    cv2.imshow('Dilated', dilated)

    # Find lines in the image
    lines = cv2.HoughLinesP(dilated, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    # Find the thinnest point on each line
    thinnest_point = None
    thinnest_thickness = None
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate average distance to non-edge pixels for thickness estimation
        dist = cv2.distanceTransform(edges, cv2.DIST_L2, maskSize=5)  # Specify mask size
        line_mask = np.zeros_like(dist)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
        non_edge_dist = dist[line_mask != 0]
        thickness = np.mean(non_edge_dist)

        # Find center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Update if thinner point found
        if thinnest_thickness is None or thickness < thinnest_thickness:
            thinnest_point = (center_x, center_y)
            thinnest_thickness = thickness

    # Draw the thinnest point and line segment
    if thinnest_point is not None:
        cv2.circle(image, thinnest_point, 5, (0, 0, 255), -1)
        

    # Display the image
   
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()








