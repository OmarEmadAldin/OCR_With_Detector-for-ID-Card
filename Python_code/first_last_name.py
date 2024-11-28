import pytesseract
from pytesseract import Output
import cv2
import matplotlib.pyplot as plt

def process_name_labels(img, annotations, lang='ara'):
    """
    Processes the labels 'firstName' and 'lastName' from annotations.
    Extracts the region of interest (ROI) from the image, preprocesses it for OCR,
    and returns the first and second names separately.

    :param img: The image from which to extract the region.
    :param annotations: Dictionary containing the annotations for the image.
    :param lang: Language for OCR (default is 'ara' for Arabic).
    :return: The extracted first name and last name.
    """
    
    first_name = None
    second_name = None

    # Process the labels 'firstName' and 'lastName'
    for label in ['firstName', 'lastName']:
        if label in annotations:
            for item in annotations[label]:
                x1, y1, x2, y2 = item['box']
                # Extract the region from the image
                roi = img[y1:y2, x1:x2]

                # Preprocess the region to improve OCR accuracy
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Resize or wrap the region if necessary
                if label == 'firstName':
                    desired_size = (2480, 1680)  # Size for firstName
                    wrapped_roi = cv2.resize(binary_roi, desired_size, interpolation=cv2.INTER_LINEAR)
                    
                    # Perform OCR with specific config for firstName
                    config = '--oem 1 --psm 7'
                    first_name = pytesseract.image_to_string(wrapped_roi, lang=lang, config=config).strip()
                else:
                    desired_size = (1240, 300)  # Size for lastName
                    wrapped_roi = cv2.resize(binary_roi, desired_size, interpolation=cv2.INTER_LINEAR)
                    
                    # Perform OCR without specific config for lastName
                    second_name = pytesseract.image_to_string(wrapped_roi, lang=lang).strip()

    # Return first name and second name separately
    return first_name, second_name

