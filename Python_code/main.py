from ultralytics import YOLO
import cv2
from check_gpu import check_device
from face import process_photo_label
from first_last_name import process_name_labels
from National_id import extract_nid_info
from religion_gender_maritalState import extract_text_from_bounding_boxes
from Job_info import extract_job_text_from_bounding_boxes
from detector import process_images_with_model
from address import process_address_label

from fillpdf import fillpdfs

# Load images
front_img = cv2.imread("/home/omar_ben_emad/ID_Card_Data_Extraction/Images/testtttttt.jpg")
back_img = cv2.imread("/home/omar_ben_emad/ID_Card_Data_Extraction/Images/fdd.jpg")

# Load model path
model_path = "/home/omar_ben_emad/ID_Card_Data_Extraction/Egyptian_id_data_extractor/runs/detect/train3/weights/best.pt"

# Check for GPU availability
check_device()

# Process both images and get the annotations
front_annotations, back_annotations = process_images_with_model(front_img, back_img, model_path)

# Print the annotations to verify
print("Front Image Annotations:", front_annotations)
print("Back Image Annotations:", back_annotations)

# First and last name 
first_name, second_name = process_name_labels(front_img, front_annotations)
print("First Name:", first_name)
print("Second Name:", second_name)

# National ID , birth_year , birth_month , birth_day
nid_info = extract_nid_info(front_img, front_annotations)

national_id = nid_info['National ID']
governorate = nid_info['Governorate']
birth_year = nid_info['Birth Year']
birth_month = nid_info['Birth Month']
birth_day = nid_info['Birth Day']

# Print the extracted information
print(f"National ID: {national_id}")
print(f"Birth Year: {birth_year}")
print(f"Birth Month: {birth_month}")
print(f"Birth Day: {birth_day}")
print(f"Governorate: {governorate}")

# Address
address = process_address_label(front_img, front_annotations)

# Job Data Extraction
job_text = extract_job_text_from_bounding_boxes(back_img, back_annotations)

# Religion, gender, and marital status 
states = extract_text_from_bounding_boxes(back_img, back_annotations)

# Face
process_photo_label(front_img, front_annotations)

# Get form fields
form_fields = list(fillpdfs.get_form_fields("/home/omar_ben_emad/ID_Card_Data_Extraction/Python_code/National_ID_Data.pdf").keys())
print(form_fields)

#Arabic Fields

# Fill data into the PDF
data_dict = {
    form_fields[0]: first_name,   # First Name
    form_fields[1]: second_name,   # Second Name
    form_fields[3]: address,       # Address
    form_fields[4]: states,        # Religion, Gender, Marital State
    form_fields[2]: national_id,   # National ID
    form_fields[5]: birth_year,    # Birth Year
    form_fields[6]: birth_month,   # Birth Month
    form_fields[7]: birth_day,     # Birth Day
    form_fields[8]: governorate,
    form_fields[9]: job_text,      # Job

}

print("Data Dictionary:", data_dict)

# Fill the PDF with the data
input_pdf_path = "/home/omar_ben_emad/ID_Card_Data_Extraction/Python_code/National_ID_Data.pdf"
output_pdf_path = 'filled_id_card.pdf'

try:
    fillpdfs.write_fillable_pdf(
        input_pdf_path=input_pdf_path,
        output_pdf_path=output_pdf_path,
        data_dict=data_dict
    )
    fillpdfs.place_image('/home/omar_ben_emad/ID_Card_Data_Extraction/Images/photo_bbox.png', 375, 50, 'filled_id_card.pdf', 'filled_id_cardwithimage.pdf', 1, width=175, height=175)
except Exception as e:
    print("Error while filling the PDF:", e)


