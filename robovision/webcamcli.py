import cv2
import math
from robovision import Robovision
from robovision.utils import class_names, class_width
from networktables import NetworkTables

# Initialize NetworkTables
NetworkTables.initialize(server='localhost')
table = NetworkTables.getTable('Vision')

# Initialize the webcam
cap = cv2.VideoCapture(0)

model_path = "models\\best.onnx"
robovision = Robovision(model_path, conf_thres=0.3, iou_thres=0.3)

# List to store object information
objects = []

focal_length = 600  # Example value, replace with your actual value

# Define class names
while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids, masks = robovision(frame)

    # Get the dimensions of the frame
    frame_height, frame_width, _ = frame.shape

    # Calculate dynamic center coordinates
    center_x = frame_width // 2
    center_y = frame_height // 2

    # Process object information
    objects = []  # Clear the list for each frame
    for i, box in enumerate(boxes):
        class_id = class_ids[i]

        class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

        # Get the known width based on class ID
        known_width = class_width[class_id] if class_id < len(class_width) else 0.2  # Default to 0.2 if not found

        # Calculate object width in pixels
        object_width_pixels = box[2] - box[0]

        # Calculate distance
        distance = (known_width * focal_length) / object_width_pixels

        # Calculate angle from the center of the object to the center of the frame
        object_center_x = (box[0] + box[2]) / 2
        object_center_y = (box[1] + box[3]) / 2

        delta_x = object_center_x - center_x
        delta_y = object_center_y - center_y

        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)

        # Append the object information to the list
        objects.append({
            'class_name': class_name,
            'distance': distance,
            'angle': angle_deg
        })

    # Sort objects by distance
    objects.sort(key=lambda x: x['distance'])

    # Update NetworkTables
    for obj in objects:
        table.putNumber(f"{obj['class_name']}_Distance", obj['distance'])
        table.putNumber(f"{obj['class_name']}_Angle", obj['angle'])

        # Print the values in real-time
        print(f"{obj['class_name']}: Distance={obj['distance']:.2f} meters, Angle={obj['angle']:.2f} degrees")

# Release resources
cap.release()
