import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def process_images_with_model(front_img, back_img, model_path):
    # Load the YOLO model
    model = YOLO(model_path)

    # Initialize dictionaries to store annotations for both images
    front_face_annotations = {}
    back_face_annotations = {}

    # Function to process a single image and return annotations
    def process_single_image(img, annotation_dict, draw_confidence=True):
        results = model(img)

        # Define colors and styles for drawing
        box_color = (0, 255, 255)  # Yellow color in BGR
        text_color = (255, 255, 255)  # White color in BGR
        text_background_color = (0, 0, 0)  # Black background in BGR
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.8
        font_thickness = 2
        box_thickness = 3

        # Process the model's results
        for result in results:
            boxes = result.boxes.xyxy  # Bounding boxes (x1, y1, x2, y2)
            confidences = result.boxes.conf  # Confidence scores
            labels = result.boxes.cls  # Class labels

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                label = labels[i]
                confidence = confidences[i]
                class_name = model.names[int(label)]

                # Store the bounding box in the appropriate dictionary
                if class_name not in annotation_dict:
                    annotation_dict[class_name] = []
                annotation_dict[class_name].append({
                    'box': (x1, y1, x2, y2),
                })

                # Print the class name and bounding box details
                print(f"Class: {class_name}, Confidence: {confidence:.2f}")
                print(f"Bounding Box: {x1}, {y1}, {x2}, {y2}")

                # Draw the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thickness)

                # Draw the label and confidence with text background if required
                if draw_confidence:
                    text = f'{class_name} {confidence:.2f}'
                    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                    text_w, text_h = text_size
                    cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), text_background_color, cv2.FILLED)
                    cv2.putText(img, text, (x1, y1 - 5), font, font_scale, text_color, font_thickness)

        # Convert BGR image (OpenCV default) to RGB for matplotlib display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Plot the image with bounding boxes
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.axis('off')  # Hide the axis
        plt.show()

    # Process the front image and update front_face_annotations
    process_single_image(front_img, front_face_annotations, draw_confidence=True)

    # Process the back image and update back_face_annotations without drawing confidence
    process_single_image(back_img, back_face_annotations, draw_confidence=False)

    # Return the annotations for both images
    return front_face_annotations, back_face_annotations
