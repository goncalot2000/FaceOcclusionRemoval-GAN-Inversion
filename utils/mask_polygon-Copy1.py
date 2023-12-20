import cv2
import numpy as np

"""
# Load the image
image_mask = cv2.imread('Result/NatOcc_hand_sot/occlusion_mask/Recep_Tayyip_Erdogan_0020_01.png', 0)
img = cv2.imread('dataset/Recep/Recep_Tayyip_Erdogan_0020_01.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
"""

def get_maks_percentage(img, image_mask, save_image_flag=False):
    
    #print(f"Image dtype: {img.dtype}")
    
    # Verify image channels
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input image should have shape [height, width, channel] with 3 channels (RGB)!")
    
    # Verify image depth (CV_8U)
    if img.dtype != np.uint8:
        # Convert image data type to numpy.uint8
        #img = img.astype(np.uint8)
        raise ValueError("Input image should have dtype `numpy.uint8`!")
        
    # Detect faces in the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    #print(f"Scale Factor: {face_classifier.scaleFactor}")
    
    face = face_classifier.detectMultiScale(gray)
    #print(len(face))
    
    if(len(face) == 0):
        covered_percentage = 100
        save_image_flag = False
        
        print(f"No face identified.")
        
    else:
        for (x, y, w, h) in face:
            img_rect = [x, y, w, h]

        # Threshold the image to convert it to a binary image
        _, binary_image = cv2.threshold(image_mask, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area (assuming it represents the white figure)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a blank image of the same size as the original image
        output_image = np.zeros_like(image_mask)

        # Draw the rectangle on the output image
        #x, y, w, h = 200, 200, 400, 500
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

    print(f"Covered area percentage: {covered_percentage:.2f}%")
    
    if save_image_flag:
        # Draw the rectangle contour on the output image
        cv2.drawContours(output_image, [largest_contour], -1, 255, 2)

        cv2.rectangle(intersection, pt1=(img_rect[0], img_rect[1]), pt2=(img_rect[0] + img_rect[2], img_rect[1] + img_rect[3]), color=(255,0,0), thickness=10)

        # Save the intersection image
        #cv2.imwrite("intersection_image.png", intersection)       
    else:
        intersection = -1
        
    return covered_percentage, intersection