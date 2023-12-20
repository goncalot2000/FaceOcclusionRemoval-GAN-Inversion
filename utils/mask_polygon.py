import cv2
import numpy as np

def get_mask_percentage(img, image_mask, G_resolution, save_image_flag=False):
    """Gets the percentage of face pixels occupied by the oclusion.
    
  Args:
    img: Pre processed image [G_resolution, G_resolution]
    image_mask: Binary mask representing the oclusion
    G_resolution: Resolution for the images of the current used generator
    save_image_flag: Whether to save an image representing the rectangle of the face with the oclusion inside.
      (default: False)

  Returns:
    The percentage of face pixels occupied by the oclusion and an image representing the rectangle of the face with the oclusion inside.
  """
    
    image_mask = cv2.resize(image_mask, (G_resolution, G_resolution))
    
    # Verify image channels
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input image should have shape [height, width, channel] with 3 channels (RGB)!")
    
    # Verify image depth (CV_8U)
    if img.dtype != np.uint8:
        # Convert image data type to numpy.uint8
        #img = img.astype(np.uint8)
        raise ValueError("Input image should have dtype `numpy.uint8`!")       

    img_rect = [int(G_resolution/5), int(G_resolution/5), int(G_resolution/1.5), int(G_resolution/1.5)]
        
    # Threshold the image to convert it to a binary image
    _, binary_image = cv2.threshold(image_mask, 1, 255, cv2.THRESH_BINARY)

    # Create a blank image of the same size as the original image
    output_image = np.zeros_like(image_mask)

    # Draw the rectangle on the output image
    cv2.rectangle(output_image, (img_rect[0], img_rect[1]), (img_rect[0] + img_rect[2], img_rect[1] + img_rect[3]), 255, -1)

    # Calculate the intersection between the rectangle and the binary mask image
    intersection = cv2.bitwise_and(output_image, binary_image)

    # Calculate the total area of the rectangle
    total_area = img_rect[2] * img_rect[3]

    # Calculate the covered area
    covered_area = np.sum(intersection == 255)

    # Calculate the percentage of covered area
    covered_percentage = (covered_area / total_area) * 100

    if(covered_percentage > 100): covered_percentage = 100

    #print(f"Covered area percentage: {covered_percentage:.2f}%")

    if save_image_flag:
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area (assuming it represents the white figure)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw the rectangle contour on the output image
        cv2.drawContours(output_image, [largest_contour], -1, 255, 2)

        cv2.rectangle(intersection, pt1=(img_rect[0], img_rect[1]), pt2=(img_rect[0] + img_rect[2], img_rect[1] + img_rect[3]), color=(255,0,0), thickness=10)

        # Save the intersection image
        #cv2.imwrite("intersection_image.png", intersection)       
    else:
        intersection = -1

    return covered_percentage, intersection