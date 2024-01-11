import cv2
import math
from robovision import Robovision
from robovision.utils import class_names, class_width

# Initialize the webcam
cap = cv2.VideoCapture(0)

model_path = "robovision\\models\\best.onnx"
robovision = Robovision(model_path, conf_thres=0.3, iou_thres=0.3)

# List to store object widths and angles
object_widths = []
object_angles = []

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

    # Draw bounding boxes, labels, distance, and angle information on the frame
    for i, box in enumerate(boxes):
        class_id = class_ids[i]

        class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

        # Draw bounding box
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # Display class name and score
        label = f"{class_name}: {scores[i]:.2f}"
        cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Get the known width based on class ID
        known_width = class_width[class_id] if class_id < len(class_width) else 0.2  # Default to 0.2 if not found

        # Calculate object width in pixels
        object_width_pixels = box[2] - box[0]

        # Calculate distance
        distance = (known_width * focal_length) / object_width_pixels
        distance_text = f"Distance: {distance:.2f} meters"
        cv2.putText(frame, distance_text, (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate angle from the center of the object to the center of the frame
        object_center_x = (box[0] + box[2]) / 2
        object_center_y = (box[1] + box[3]) / 2

        delta_x = object_center_x - center_x
        delta_y = object_center_y - center_y

        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)

        # Display angle information
        angle_text = f"Angle: {angle_deg:.2f} degrees"
        cv2.putText(frame, angle_text, (int(box[0]), int(box[1]) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Append the object width and angle to the lists
        object_widths.append(object_width_pixels)
        object_angles.append(angle_deg)

    combined_img = robovision.draw_masks(frame)
    # Display the frame in a window
    cv2.imshow('Object Detection', combined_img)

    # Press key q to stop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the list of object widths and angles
print("Object Widths (in pixels):", object_widths)
print("Object Angles (in degrees):", object_angles)
