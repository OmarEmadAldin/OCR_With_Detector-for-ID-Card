# ID_Scanner_Data_Extractor
Extract information from ID cards using Optical Character Recognition (OCR) and pre-defined Regions of Interest (ROIs). This project automates data extraction and fills it into structured files like PDFs or databases.

## **Purpose**

The main goal of this project is to:
- Automate the process of extracting specific information from ID cards.
- Accurately populate extracted data into structured output formats (e.g., PDF forms ,JSON , DOCX).
- Minimize manual data entry efforts in administrative and data-heavy workflows.which would minimize the time and effort


## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - OpenCV (for image processing)
  - PyTesseract (for OCR)
  - Ultralytics YOLOv8 (for detecting ROIs)
  - fillpdf (for PDF population)

- **Frameworks**: Flask (if applicable for web hosting)
- **File Formats**: JSON, PDF, Image Formats (JPG, PNG)



## **Requirements**
Ensure you have the following installed:
- Python 3, can be any higher version and 
- Tesseract OCR (Ensure it is installed and added to PATH). that will be applied for the file [ara_number_id.traineddata]. to have the ability to extract arabic data

- Python Libraries:
  ```bash
    pip install opencv-python-headless ultralytics fillpdf pytesseract matplotlib
    pip install torch torchvision torchaudio
    ```

## **Files**

- Egyptian_id_data_extractor: 
This File is related to the training of YOLO Detectort to get [best.pt] file, which later i use it with loading the model.

- dataset.zip:
Contains the dataset with its data.yaml file

- Python_code:
Contains all the python files from the detection till the file generation

- filled_id_card.pdf:
Fillable pdf file which we use to fill it with the data extracted from the code

- filled_id_cardwithimage.pdf:
Result file that have the pdf file with the extracted data. (Opens well in browser pdf viewer not in adobe pdf, better to use another file format like JSON or DOCX)

- ara_number_id.traineddata
The Training file used to deploy arabic OCR

## **Usage**

Just Run the [main.py] file as it calls all the functions and each function do a specific task which is obvious from the python file name

## **Steps**
### **1. Load Images**
Load the front and back images of the ID card using OpenCV:
```python
front_img = cv2.imread("/path/to/front_image.jpg")
back_img = cv2.imread("/path/to/back_image.jpg")
```
### **2. Load Images**

Specify the path to the YOLO model trained for ID card processing:
```python
model_path = "/path/to/yolo_model/best.pt"
```
### **3. Check GPU Availability**

Use the check_gpu function which is in the file Python_code in [check_gpu.py] to ensure the GPU is available for processing:
```python
check_device()
```

### **4. Process Images and Extract Annotations**
Use the YOLO model to process the front and back images, returning bounding box annotations, we return it as a dictionary and the keywords are the labels and values are the (x_1 , x_2 , y_1 , y_2) of the bounding box of each class.


## **5. Extract Information**

### **5.1 Names**
- Extract the first and second names from the front image annotations.

---

### **5.2 National ID and Related Details**
- Extract the following information from the front image annotations:
  - National ID
  - Birth year, month, and day
  - Governorate

---

### **5.3 Address**
- Extract the address details from the front image.

---

### **5.4 Job Information**
- Extract job-related text from the back image.

---

### **5.5 Religion, Gender, and Marital Status**
- Extract religion, gender, and marital status from the back image.

---

### **5.6 Face Processing**
- Process and extract the photo from the front image.

---

## **6. Prepare the PDF Form**
1. Retrieve the list of fillable fields from the provided PDF template.
2. Map the extracted data to the corresponding PDF fields, including:
   - First Name
   - Second Name
   - Address
   - Religion, Gender, and Marital Status
   - National ID
   - Birth Year, Month, and Day
   - Governorate
   - Job Information

---

## **7. Fill the PDF Form**
1. Use the extracted information to populate the fields in the PDF form.
2. Add the processed photo to the PDF, placing it in the designated location.
