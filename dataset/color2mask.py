import cv2
import numpy as np
import os
import random

def maskColorObject(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lowerRedOrange = np.array([0, 150, 100])
    upperRedOrange = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lowerRedOrange, upperRedOrange)

    colorObject = cv2.bitwise_and(image, image, mask=mask)

    return colorObject, mask

def filterContours(contours, minArea=6000):
    filteredContours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea:
            filteredContours.append(contour)
    return filteredContours

def savePolygonToTxt(polygon, filepath, imageWidth, imageHeight):
    with open(filepath, 'w') as file:
        file.write("0 ")
        for point in polygon:
            x, y = point

            normalizedX = x / imageWidth
            normalizedY = y / imageHeight

            file.write(f"{normalizedX} {normalizedY} ")

        file.write('\n')

def maskImageBasedOnPolygon(image, polygon):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [polygon], (255, 255, 255)) 
    maskedImage = cv2.bitwise_and(image, mask)
    return maskedImage

def splitDataset(imagesFolder, trainRatio=0.7, valRatio=0.2, testRatio=0.1):
    imageFiles = [file for file in os.listdir(imagesFolder) if file.endswith(('.jpg', '.jpeg', '.png'))]

    random.shuffle(imageFiles)

    totalImages = len(imageFiles)
    numTrain = int(totalImages * trainRatio)
    numVal = int(totalImages * valRatio)
    numTest = totalImages - numTrain - numVal

    trainFiles = imageFiles[:numTrain]
    valFiles = imageFiles[numTrain:numTrain+numVal]
    testFiles = imageFiles[numTrain+numVal:]

    return trainFiles, valFiles, testFiles

def createDatasetFolders(baseFolder):
    datasetFolder = os.path.join(baseFolder, 'images')
    trainFolder = os.path.join(datasetFolder, 'train')
    valFolder = os.path.join(datasetFolder, 'val')
    testFolder = os.path.join(datasetFolder, 'test')

    for folder in [trainFolder, valFolder, testFolder]:
        os.makedirs(folder, exist_ok=True)

    return trainFolder, valFolder, testFolder

def processImages(inputFolder, outputFolder, labelsFolder):
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for root, dirs, files in os.walk(inputFolder):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                imagePath = os.path.join(root, filename)
                image = cv2.imread(imagePath)

                _, mask = maskColorObject(image)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                filteredContours = filterContours(contours)
                for idx, contour in enumerate(filteredContours):
                    polygon = contour.squeeze()

                    filenameWithoutExtension, _ = os.path.splitext(filename)
                    txtFilename = f"{filenameWithoutExtension}.txt"
                    txtFilepath = os.path.join(labelsFolder, txtFilename)

                    imageWidth, imageHeight = image.shape[1], image.shape[0]

                    savePolygonToTxt(polygon, txtFilepath, imageWidth, imageHeight)

                    maskedImage = maskImageBasedOnPolygon(image, polygon)
                    outputImagePath = os.path.join(outputFolder, filename)
                    outputMaskImagePath = os.path.join(outputFolder, "mask_" + filename)
                    cv2.imwrite(outputImagePath, image)
                    cv2.imwrite(outputMaskImagePath, maskedImage)

if __name__ == "__main__":
    inputFolder = "images"
    outputFolder = "output"

    trainFolder, valFolder, testFolder = createDatasetFolders(outputFolder)

    trainLabelsFolder = os.path.join(outputFolder, 'labels', 'train')
    valLabelsFolder = os.path.join(outputFolder, 'labels', 'val')
    testLabelsFolder = os.path.join(outputFolder, 'labels', 'test')

    for folder in [trainLabelsFolder, valLabelsFolder, testLabelsFolder]:
        os.makedirs(folder, exist_ok=True)

    trainFiles, valFiles, testFiles = splitDataset(inputFolder)

    processImages(inputFolder, trainFolder, trainLabelsFolder)
    processImages(inputFolder, valFolder, valLabelsFolder)
    processImages(inputFolder, testFolder, testLabelsFolder)
