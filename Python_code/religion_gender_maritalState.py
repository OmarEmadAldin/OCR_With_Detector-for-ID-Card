import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

def extract_text_from_bounding_boxes(img, annotations, label='demo', scale_factor=2):
    """
    Extracts and processes text from specified bounding boxes in an image using OCR.

    Parameters:
    - img: The input image containing the regions of interest.
    - annotations: The annotations containing bounding box information.
    - label: The specific label to look for in annotations.
    - scale_factor: Factor to resize the ROI for improved OCR accuracy.

    Returns:
    - extracted_text: The extracted text from the specified bounding boxes, concatenated as a string with spaces between words.
    """
    extracted_text = ""  # Initialize an empty string to store extracted text

    # Check if the specified label is in the annotations
    if label in annotations:
        for item in annotations[label]:
            x1, y1, x2, y2 = item['box']
            # Extract the region from the image
            roi = img[y1:y2, x1:x2]
            print(np.shape(roi))

            # Preprocess the region to improve OCR accuracy
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dilated_roi = cv2.dilate(binary_roi, np.ones((2, 2), np.uint8), iterations=1)  # Thicken the text

            # Resize the ROI to improve OCR accuracy
            resized_roi = cv2.resize(dilated_roi, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

            # Display the extracted and processed region (Optional)
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(resized_roi, cv2.COLOR_GRAY2RGB))  # Adjusted for grayscale image
            plt.title(f'{label} - Resized and Processed Bounding Box')
            plt.axis('off')  # Hide the axis
            plt.show()

            # Perform OCR with specific config for general text recognition
            custom_config_text = r'--oem 3 --psm 6 -l ara'
            text = pytesseract.image_to_string(resized_roi, lang='ara', config=custom_config_text)
            print(f'{label} - Corrected OCR Text: {text}')

            # Append the extracted text to the string, separating words with a space
            extracted_text += text.strip() + " "  # Remove leading/trailing whitespace and add a space

    # Return the final extracted text with extra spaces trimmed
    return extracted_text.strip()
