import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import re  # For cleaning the text

def extract_job_text_from_bounding_boxes(img, annotations, label='job'):
    """
    Extracts and cleans text from specified bounding boxes in an image using OCR for job labels.

    Parameters:
    - img: The input image containing the regions of interest.
    - annotations: The annotations containing bounding box information.
    - label: The specific label to look for in annotations.

    Returns:
    - text: Cleaned and extracted text from the specified bounding boxes.
    """
    extracted_text = ""

    # Loop through each label and its corresponding bounding box
    if label in annotations:
        for item in annotations[label]:
            x1, y1, x2, y2 = item['box']
            
            # Extract the region from the image
            roi = img[y1:y2, x1:x2]
            print(f'Shape of ROI for {label}: {np.shape(roi)}')

            # Convert ROI to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Display the extracted region
            plt.figure(figsize=(6, 6))
            plt.imshow(roi)  # Display the original ROI
            plt.title(f'{label} - Extracted Bounding Box')
            plt.axis('off')  # Hide the axis
            plt.show()

            # Perform OCR with specific config for Arabic text
            custom_config_text = r'--oem 3 --psm 6 -l ara'
            text = pytesseract.image_to_string(gray_roi, config=custom_config_text).strip()

            # Clean the extracted text by removing special characters and undefined letters
            cleaned_text = re.sub(r'[^ء-ي\s]', '', text)  # Keep only Arabic letters and spaces

            # Append the cleaned text
            extracted_text += f"{cleaned_text}\n"  # Use newline for separating text from multiple boxes
            
            print(f'{label} - OCR Text: {text}')
            print(f'{label} - Cleaned Text: {cleaned_text}')

    return extracted_text.strip()  # Return the cleaned and extracted text
