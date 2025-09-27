import cv2
import math
from robovision import Robovision
from robovision.utils import classNames, classWidth

cap = cv2.VideoCapture(0)

modelPath = "models\\best.onnx"
robovision = Robovision(modelPath, conf_thres=0.3, iou_thres=0.3)

objectWidths = []
objectAngles = []

focalLength = 600
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    boxes, scores, classIds, masks = robovision(frame)

    frameHeight, frameWidth, _ = frame.shape

    centerX = frameWidth // 2
    centerY = frameHeight // 2

    for i, box in enumerate(boxes):
        classId = classIds[i]

        className = classNames[classId] if classId < len(classNames) else "Unknown"

        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        label = f"{className}: {scores[i]:.2f}"
        cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        knownWidth = classWidth[classId] if classId < len(classWidth) else 0.2

        objectWidthPixels = box[2] - box[0]

        distance = (knownWidth * focalLength) / objectWidthPixels
        distanceText = f"Distance: {distance:.2f} meters"
        cv2.putText(frame, distanceText, (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        objectCenterX = (box[0] + box[2]) / 2
        objectCenterY = (box[1] + box[3]) / 2

        deltaX = objectCenterX - centerX
        deltaY = objectCenterY - centerY

        angleRad = math.atan2(deltaY, deltaX)
        angleDeg = math.degrees(angleRad)

        angleText = f"Angle: {angleDeg:.2f} degrees"
        cv2.putText(frame, angleText, (int(box[0]), int(box[1]) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        objectWidths.append(objectWidthPixels)
        objectAngles.append(angleDeg)

    combinedImg = robovision.drawMasks(frame)
    cv2.imshow('Object Detection', combinedImg)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Object Widths (in pixels):", objectWidths)
print("Object Angles (in degrees):", objectAngles)
