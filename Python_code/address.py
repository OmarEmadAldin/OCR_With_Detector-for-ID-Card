import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import re

def process_address_label(img, annotations, label='address', lang='ara'):
    """
    Processes the specified label (default is 'address') from annotations.
    Extracts the region of interest (ROI) from the image, preprocesses it for OCR,
    and returns the recognized address text.

    :param img: The image from which to extract the region.
    :param annotations: Dictionary containing the annotations for the image.
    :param label: The label to process (default is 'address').
    :param lang: Language for OCR (default is 'ara' for Arabic).
    :return: The extracted address text from the image.
    """
    address_text = ""
    
    if label in annotations:
        for item in annotations[label]:
            x1, y1, x2, y2 = item['box']
            
            # Extract the region from the image
            roi = img[y1:y2, x1:x2]
            print(f"ROI shape for {label}: {np.shape(roi)}")

            # Preprocess the region to improve OCR accuracy
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Additional preprocessing (optional)
            denoised_roi = cv2.medianBlur(binary_roi, 3)  # Reduces noise

            # Display the extracted and processed region
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(denoised_roi, cv2.COLOR_GRAY2RGB))  # Adjusted for grayscale
            plt.title(f'{label.capitalize()} - Bounding Box')
            plt.axis('off')  # Hide the axis
            plt.show()

            # Perform OCR with specific config for general text recognition
            custom_config_text = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(denoised_roi, lang=lang, config=custom_config_text)
            print(f'{label.capitalize()} - OCR Raw Text: {text.strip()}')

            # Post-processing: Remove special characters and keep only valid letters/numbers
            cleaned_text = re.sub(r'[^A-Za-z0-9ء-ي ]+', '', text)  # Arabic + English alphanumeric
            
            # Strip extra spaces and save the text
            address_text += cleaned_text.strip() + " "
    
    return address_text.strip()  # Return the extracted and cleaned address text
