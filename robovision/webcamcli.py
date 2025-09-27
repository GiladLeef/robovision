import cv2
import math
from robovision import Robovision
from robovision.utils import classNames, classWidth
from networktables import NetworkTables

NetworkTables.initialize(server='roborio-1937-frc.local')
table = NetworkTables.getTable('Vision')

modelPath = "models/best.onnx"
robovision = Robovision(modelPath, conf_thres=0.3, iou_thres=0.3)
cap = cv2.VideoCapture(0)
focalLength = 600

desiredFrameRate = 5
frameDelay = 1 / desiredFrameRate

def processFrame():
    ret, frame = cap.read()

    if not ret:
        return None

    boxes, scores, classIds, masks = robovision(frame)

    frameHeight, frameWidth, _ = frame.shape

    centerX = frameWidth // 2
    centerY = frameHeight // 2

    objects = []
    for i, box in enumerate(boxes):
        classId = classIds[i]

        className = classNames[classId] if classId < len(classNames) else "Unknown"

        knownWidth = classWidth[classId] if classId < len(classWidth) else 0.2

        objectWidthPixels = box[2] - box[0]

        distance = (knownWidth * focalLength) / objectWidthPixels

        objectCenterX = (box[0] + box[2]) / 2
        objectCenterY = (box[1] + box[3]) / 2

        deltaX = objectCenterX - centerX
        deltaY = objectCenterY - centerY

        deviation = centerX - objectCenterX
        scaledDeviation = (deviation / (frameWidth // 2)) * 256

        objects.append({
            'className': className,
            'distance': distance,
            'angle': scaledDeviation
        })

        objects.sort(key=lambda x: x['distance'])

    return objects

while True:
    objects = processFrame()

    if objects is None:
        print('Failed to read frame')
    else:
        for obj in objects:
            print(f"{obj['className']}: Distance={obj['distance']:.2f} meters, Angle={obj['angle']:.2f} degrees")

        if objects:
            lowestDistanceObject = objects[0]
            table.putNumber('Distance', lowestDistanceObject['distance'])
            table.putNumber('Angle', lowestDistanceObject['angle'])
        else:
            table.putNumber('Distance', 0.0)
            table.putNumber('Angle', 0.0)
