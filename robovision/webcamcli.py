import cv2
import math
from robovision import Robovision
from robovision.utils import class_names, class_width
from networktables import NetworkTables

# Connect to the NetworkTables server on the roboRIO
NetworkTables.initialize(server='roborio-1937-frc.local')
table = NetworkTables.getTable('Vision')

model_path = "models/best.onnx"
robovision = Robovision(model_path, conf_thres=0.3, iou_thres=0.3)
cap = cv2.VideoCapture(0)
focal_length = 600

# Set the desired frame rate (in frames per second)
desired_frame_rate = 5
frame_delay = 1 / desired_frame_rate

def process_frame():
    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        return None

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

        # Calculate angle in degrees with positive values for the right and negative values for the left
        angle_deg = math.degrees(math.atan2(delta_y, delta_x))

        # Calculate the deviation from the center of the screen
        deviation = center_x - object_center_x
        scaled_deviation = (deviation / (frame_width // 2)) * 256

        # Append the object information to the list
        objects.append({
            'class_name': class_name,
            'distance': distance,
            'angle': scaled_deviation
        })

        # Sort objects by distance
        objects.sort(key=lambda x: x['distance'])

    return objects

# Define class names
while True:
    # List to store object information
    objects = process_frame()

    if objects is None:
        print('Failed to read frame')
    else:
        # Print the values in real-time
        for obj in objects:
            print(f"{obj['class_name']}: Distance={obj['distance']:.2f} meters, Angle={obj['angle']:.2f} degrees")

        # Send data to NetworkTables for the object with the lowest distance
        if objects:
            lowest_distance_object = objects[0]
            table.putNumber('Distance', lowest_distance_object['distance'])
            table.putNumber('Angle', lowest_distance_object['angle'])
        else:
            # If no objects detected, write "0.0" values to NetworkTables
            table.putNumber('Distance', 0.0)
            table.putNumber('Angle', 0.0)
