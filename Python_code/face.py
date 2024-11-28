import cv2
import matplotlib.pyplot as plt

def process_photo_label(img, annotations, label='photo', desired_size=(300, 300)):
    """
    Processes only the specified label (default is 'photo') from annotations.
    Extracts the region of interest (ROI) from the image based on bounding box coordinates,
    resizes the ROI, and displays it.

    :param img: The image from which to extract the region.
    :param annotations: Dictionary containing the annotations for the image.
    :param label: The label to process (default is 'photo').
    :param desired_size: Desired size to resize the ROI (default is (300, 300)).
    """
    if label in annotations:
        for item in annotations[label]:
            x1, y1, x2, y2 = item['box']
            
            # Extract the region from the image
            roi = img[y1:y2, x1:x2]

            # Resize the region
            wrapped_roi = cv2.resize(roi, desired_size, interpolation=cv2.INTER_LINEAR)

            # Display the extracted and processed region
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(wrapped_roi, cv2.COLOR_BGR2RGB))
            plt.title(f'{label.capitalize()} - Bounding Box')
            plt.axis('off')  # Hide the axis
            plt.show()
    else:
        print(f"No label '{label}' found in annotations.")
