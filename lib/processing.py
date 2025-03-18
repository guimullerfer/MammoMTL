import cv2
import numpy as np

def processing(gray_image, gray_annotation_image=None):
    # Convert the image to 8 bits (0-255)
    gray_image_8bits = cv2.convertScaleAbs(gray_image, alpha=(255.0/np.max(gray_image)))

    # Binarize the image
    _, mask = cv2.threshold(gray_image_8bits, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (representing the breast region)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask for the largest region found
    largest_object_mask = np.zeros_like(mask)
    cv2.fillPoly(largest_object_mask, [largest_contour], 255)
    
    # Apply the mask to remove the background from the image
    image_no_background = cv2.bitwise_and(gray_image, gray_image, mask=largest_object_mask)
    
    # Check if the breast is on the left side
    isLeft = np.sum(image_no_background[:, :image_no_background.shape[1]//2]) > np.sum(image_no_background[:, image_no_background.shape[1]//2:])
    
    if gray_annotation_image is None:
        gray_annotation_image = np.zeros(gray_image.shape, dtype=np.uint8)
    
    if not isLeft:
        # Flip the image horizontally if the breast is on the right side
        image_no_background = cv2.flip(image_no_background, 1)
        gray_annotation_image = cv2.flip(gray_annotation_image, 1)
    
    height, width = image_no_background.shape[:2]
    
    # Calculate the number of columns needed to make the image square
    required_columns = abs(height - width)
    filled_columns = np.zeros((height, required_columns), dtype=np.uint8)

    filled_columns = filled_columns[:, :, np.newaxis][:, :, 0]

    # Concatenate the images
    filled_image = np.hstack((image_no_background, filled_columns))
    filled_annotation = np.hstack((gray_annotation_image, filled_columns))
    
    return filled_image, filled_annotation