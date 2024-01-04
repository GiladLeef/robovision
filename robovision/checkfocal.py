import cv2
import math
from robovision import Robovision
from robovision.utils import class_names, class_width

# Initialize the webcam
cap = cv2.VideoCapture(0)

model_path = "models/yolov8n-seg.onnx"
robovision = Robovision(model_path, conf_thres=0.3, iou_thres=0.3)

# List to store object widths and distances for the specified class
target_class_name = "cell phone"  # Replace with your actual target class name
target_class_widths = []
target_class_distances = []

# Number of samples to collect for calibration
num_calibration_samples = 10

# Counter to keep track of collected samples
calibration_sample_count = 0

# Flag to indicate if calibration is completed
calibration_complete = False

# Define known width and distance
known_width = 0.9
known_distance = 1.0  # Distance from the camera in meters

# Define class names
while cap.isOpened() and not calibration_complete:
    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids, masks = robovision(frame)

    # Draw bounding boxes, labels, distance, and angle information on the frame
    for i, box in enumerate(boxes):
        class_id = class_ids[i]
        class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

        if class_name == target_class_name:
            # Calculate object width in pixels
            object_width_pixels = box[2] - box[0]

            # Calculate distance
            focal_length = (object_width_pixels * known_distance) / known_width

            # Append the object width and distance to the lists
            target_class_widths.append(object_width_pixels)
            target_class_distances.append(known_distance)

            # Increase the sample count
            calibration_sample_count += 1

            # Check if enough samples are collected for calibration
            if calibration_sample_count >= num_calibration_samples:
                calibration_complete = True
                break

    # Display the frame in a window
    cv2.imshow('Calibration', frame)

    # Press key q to stop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Calculate the focal length using the collected data
average_width = sum(target_class_widths) / len(target_class_widths)
average_distance = sum(target_class_distances) / len(target_class_distances)

focal_length = (average_width * average_distance) / known_width

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the calculated focal length
print("Calculated Focal Length:", focal_length)
