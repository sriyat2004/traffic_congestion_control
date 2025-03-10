import cv2
import numpy as np

def detect_potholes(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply thresholding to detect dark spots (potholes)
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours (edges)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potholes = 0
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Minimum area of potholes
            potholes += 1
    
    return potholes
