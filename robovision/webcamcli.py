import cv2
from robovision import Robovision
from robovision.utils import class_names
# Initialize the webcam
cap = cv2.VideoCapture(0)

model_path = "models/yolov8n-seg.onnx"
robovision = Robovision(model_path, conf_thres=0.3, iou_thres=0.3)

# Define class names
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids, masks = robovision(frame)

    # Print detected objects with class names
    for i, box in enumerate(boxes):
        class_name = class_names[class_ids[i]] if class_ids[i] < len(class_names) else "Unknown"
        print(f"Object {i + 1}: Class {class_name}, Score: {scores[i]}")

    # Press key q to stop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
