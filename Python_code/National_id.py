import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

# Arabic to English numeral mapping
arabic_to_english = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}

# Governorates mapping
governorates = {
    '01': 'Cairo', '02': 'Alexandria', '03': 'Port Said', '04': 'Suez',
    '11': 'Damietta', '12': 'Dakahlia', '13': 'Ash Sharqia', '14': 'Kaliobeya',
    '15': 'Kafr El - Sheikh', '16': 'Gharbia', '17': 'Monoufia', '18': 'El Beheira',
    '19': 'Ismailia', '21': 'Giza', '22': 'Beni Suef', '23': 'Fayoum', '24': 'El Menia',
    '25': 'Assiut', '26': 'Sohag', '27': 'Qena', '28': 'Aswan', '29': 'Luxor',
    '31': 'Red Sea', '32': 'New Valley', '33': 'Matrouh', '34': 'North Sinai', 
    '35': 'South Sinai', '88': 'Foreign'
}

fake_national_id_message = 'This ID Not Valid'
def extract_nid_info(img, annotations, label='nid'):
    """
    Processes the 'nid' label from annotations, extracts the National ID, converts
    Arabic numerals to English, and returns the National ID, governorate, birth year, 
    month, and day.

    :param img: The image containing the NID.
    :param annotations: The annotations that contain the bounding box for the NID.
    :param label: The label to look for in the annotations (default is 'nid').
    :return: A dictionary containing the raw National ID, birth year, birth month, birth day, and governorate.
    """
    numbers_english = ""

    if label in annotations:
        for item in annotations[label]:
            x1, y1, x2, y2 = item['box']
            roi = img[y1:y2, x1:x2]

            # Preprocess the region for OCR
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
            _, binary_roi = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            binary_roi = cv2.dilate(binary_roi, kernel, iterations=1)
            binary_roi = cv2.erode(binary_roi, kernel, iterations=1)

            # Perform OCR for numbers
            custom_config_numbers = r'--oem 3 --psm 7 -c tessedit_char_whitelist="٠١٢٣٤٥٦٧٨٩"'
            numbers = pytesseract.image_to_string(binary_roi, lang='ara_number_id', config=custom_config_numbers)

            # Clean and convert Arabic numbers to English
            clean_numbers = numbers.replace(" ", "").strip()
            numbers_english = ''.join(arabic_to_english.get(c, c) for c in clean_numbers)

            # Check if we have enough digits for a valid National ID
            if len(numbers_english) >= 14:
                formatted_numbers = numbers_english
            else:
                formatted_numbers = numbers_english

    # Extract and validate National ID details
    if len(numbers_english) >= 14:
        governorate_code = numbers_english[7:9]
        year_part = numbers_english[1:3]
        month_part = numbers_english[3:5]
        day_part = numbers_english[5:7]

        # Check governorate
        if governorate_code in governorates:
            governorate = governorates[governorate_code]
        else:
            governorate = fake_national_id_message

        # Determine century and complete birth year
        century_digit = numbers_english[0]
        if century_digit == '2':
            birth_year = f'19{year_part}'
        elif century_digit == '3':
            birth_year = f'20{year_part}'
        else:
            birth_year = 'Unknown'

        # Return the raw extracted values
        return {
            "National ID": numbers_english,
            "Governorate": governorate,
            "Birth Year": birth_year,
            "Birth Month": month_part,
            "Birth Day": day_part
        }
    else:
        # If the National ID is invalid, return a message indicating invalid data
        return {
            "National ID": fake_national_id_message,
            "Governorate": fake_national_id_message,
            "Birth Year": fake_national_id_message,
            "Birth Month": fake_national_id_message,
            "Birth Day": fake_national_id_message
        }
