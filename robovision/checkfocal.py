import cv2
import math
from robovision import Robovision
from robovision.utils import classNames, classWidth

cap = cv2.VideoCapture(1)

modelPath = "models/best.onnx"
robovision = Robovision(modelPath, conf_thres=0.3, iou_thres=0.3)

targetClassName = "note"
targetClassWidths = []
targetClassDistances = []

numCalibrationSamples = 10

calibrationSampleCount = 0

calibrationComplete = False

knownWidth = 0.3556
knownDistance = 1.0
while cap.isOpened() and not calibrationComplete:
    ret, frame = cap.read()

    if not ret:
        break

    boxes, scores, classIds, masks = robovision(frame)

    for i, box in enumerate(boxes):
        classId = classIds[i]
        className = classNames[classId] if classId < len(classNames) else "Unknown"

        if className == targetClassName:
            objectWidthPixels = box[2] - box[0]

            focalLength = (objectWidthPixels * knownDistance) / knownWidth

            targetClassWidths.append(objectWidthPixels)
            targetClassDistances.append(knownDistance)

            calibrationSampleCount += 1

            if calibrationSampleCount >= numCalibrationSamples:
                calibrationComplete = True
                break

averageWidth = sum(targetClassWidths) / len(targetClassWidths)
averageDistance = sum(targetClassDistances) / len(targetClassDistances)

focalLength = (averageWidth * averageDistance) / knownWidth

cap.release()

print("Calculated Focal Length:", focalLength)
