import cv2
import math
from robovision import Robovision
from robovision.utils import classNames, classWidth
from networktables import NetworkTables

NetworkTables.initialize(server='roborio-1937-frc.local')
table = NetworkTables.getTable('Vision')

modelPath = "models/best.onnx"
robovision = Robovision(modelPath, conf_thres=0.3, iou_thres=0.3)

leftCamera = cv2.VideoCapture(0)
rightCamera = cv2.VideoCapture(1)

focalLength = 600

desiredFrameRate = 15
frameDelay = 1 / desiredFrameRate

def processFrame(camera):
    ret, frame = camera.read()

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

        angleDeg = math.degrees(math.atan2(deltaY, deltaX))

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
    leftObjects = processFrame(leftCamera)

    if leftObjects is None:
        print('Failed to read frame from left camera')
    else:
        for obj in leftObjects:
            print(f"Left {obj['className']}: Distance={obj['distance']:.2f} meters, Angle={obj['angle']:.2f} degrees")

        if leftObjects:
            lowestDistanceObject = leftObjects[0]
            table.putNumber('LeftDistance', lowestDistanceObject['distance'])
            table.putNumber('LeftAngle', lowestDistanceObject['angle'])
        else:
            table.putNumber('LeftDistance', 0.0)
            table.putNumber('LeftAngle', 0.0)

    rightObjects = processFrame(rightCamera)

    if rightObjects is None:
        print('Failed to read frame from right camera')
    else:
        for obj in rightObjects:
            print(f"Right {obj['className']}: Distance={obj['distance']:.2f} meters, Angle={obj['angle']:.2f} degrees")

        if rightObjects:
            lowestDistanceObject = rightObjects[0]
            table.putNumber('RightDistance', lowestDistanceObject['distance'])
            table.putNumber('RightAngle', lowestDistanceObject['angle'])
        else:
            table.putNumber('RightDistance', 0.0)
            table.putNumber('RightAngle', 0.0)
