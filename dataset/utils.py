import base64
import glob
import io
import json
import math
import os
import random
import shutil
import uuid
import logging

import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import cv2
import numpy as np
import tqdm

random.seed(12345678)
random.Random().seed(12345678)
np.random.seed(12345678)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datatool")


def trainTestSplit(datasetIndex, testSize=0.2):
    testSize = min(max(0.0, testSize), 1.0)
    totalSize = len(datasetIndex)
    trainSize = int(math.ceil(totalSize * (1.0 - testSize)))
    random.shuffle(datasetIndex)
    trainIndex = datasetIndex[:trainSize]
    testIndex = datasetIndex[trainSize:]

    return trainIndex, testIndex


def imgDataToPil(imgData):
    file = io.BytesIO()
    file.write(imgData)
    imgPil = PIL.Image.open(file)
    return imgPil


def imgDataToArr(imgData):
    imgPil = imgDataToPil(imgData)
    imgArr = np.array(imgPil)
    return imgArr


def imgB64ToArr(imgB64):
    imgData = base64.b64decode(imgB64)
    imgArr = imgDataToArr(imgData)
    return imgArr


def imgPilToData(imgPil):
    file = io.BytesIO()
    imgPil.save(file, format="PNG")
    imgData = file.getvalue()
    return imgData


def imgArrToB64(imgArr):
    imgPil = PIL.Image.fromarray(imgArr)
    file = io.BytesIO()
    imgPil.save(file, format="PNG")
    imgBin = file.getvalue()
    imgB64 = base64.encodebytes(imgBin)
    return imgB64


def imgDataToPngData(imgData):
    with io.BytesIO() as fOut:
        fOut.write(imgData)
        img = PIL.Image.open(fOut)

        with io.BytesIO() as fIn:
            img.save(fIn, "PNG")
            fIn.seek(0)
            return fIn.read()


def extendPointList(pointList, outFormat="polygon"):
    xMin = min(float(point) for point in pointList[::2])
    xMax = max(float(point) for point in pointList[::2])
    yMin = min(float(point) for point in pointList[1::2])
    yMax = max(float(point) for point in pointList[1::2])

    if outFormat == "bbox":
        xI = xMin
        yI = yMin
        wI = xMax - xMin
        hI = yMax - yMin
        xI = xI + wI / 2
        yI = yI + hI / 2
        return np.array([xI, yI, wI, hI])

    return np.array([xMin, yMin, xMax, yMin, xMax, yMax, xMin, yMax])


def saveYoloLabel(objList, labelDir, targetDir, targetName):
    txtPath = os.path.join(labelDir, targetDir, targetName)

    with open(txtPath, "w+", encoding="utf-8") as file:
        for label, points in objList:
            points = [str(item) for item in points]
            line = f"{label} {' '.join(points)}\n"
            file.write(line)


def saveYoloImage(jsonData, jsonDir, imageDir, targetDir, targetName):
    imgPath = os.path.join(imageDir, targetDir, targetName)

    if jsonData["imageData"] is None:
        imageName = jsonData["imagePath"]
        srcImageName = os.path.join(jsonDir, imageName)
        srcImage = cv2.imread(srcImageName)
        cv2.imwrite(imgPath, srcImage)
    else:
        img = imgB64ToArr(jsonData["imageData"])
        PIL.Image.fromarray(img).save(imgPath)

    return imgPath


class Datatool:

    def __init__(self, jsonDir, outputFormat, labelList):
        self._jsonDir = os.path.expanduser(jsonDir)
        self._outputFormat = outputFormat
        self._labelList = []
        self._labelIdMap = {}
        self._labelDirPath = ""
        self._imageDirPath = ""

        if labelList:
            self._labelList = labelList
            self._labelIdMap = {
                label: labelId for labelId, label in enumerate(labelList)
            }

    def _updateIdMap(self, label: str):
        if label not in self._labelList:
            self._labelList.append(label)
            self._labelIdMap[label] = len(self._labelIdMap)

    def _makeTrainValDir(self):
        self._labelDirPath = os.path.join(self._jsonDir, "dataset/labels/")
        self._imageDirPath = os.path.join(self._jsonDir, "dataset/images/")

        for yoloPath in (
            os.path.join(self._labelDirPath + "train/"),
            os.path.join(self._labelDirPath + "val/"),
            os.path.join(self._labelDirPath + "test/"),
            os.path.join(self._imageDirPath + "train/"),
            os.path.join(self._imageDirPath + "val/"),
            os.path.join(self._imageDirPath + "test/"),
        ):
            if os.path.exists(yoloPath):
                shutil.rmtree(yoloPath)

            os.makedirs(yoloPath)

    def _getDatasetPartJsonNames(self, datasetPart: str):
        setFolder = os.path.join(self._jsonDir, datasetPart)
        jsonNames = []
        for sampleName in os.listdir(setFolder):
            setDir = os.path.join(setFolder, sampleName)
            if os.path.isdir(setDir):
                jsonNames.append(sampleName + ".json")
        return jsonNames

    def _trainTestSplit(self, jsonNames, valSize, testSize):
        totalSize = len(jsonNames)
        datasetIndex = list(range(totalSize))
        trainIds, valIds = trainTestSplit(datasetIndex, testSize=valSize)
        testIds = []
        if testSize is None:
            testSize = 0.0
        if testSize > 1e-8:
            trainIds, testIds = trainTestSplit(
                trainIds, testSize=testSize / (1 - valSize)
            )
        trainJsonNames = [jsonNames[trainIdx] for trainIdx in trainIds]
        valJsonNames = [jsonNames[valIdx] for valIdx in valIds]
        testJsonNames = [jsonNames[testIdx] for testIdx in testIds]

        return trainJsonNames, valJsonNames, testJsonNames

    def convert(self, valSize, testSize):
        jsonNames = glob.glob(
            os.path.join(self._jsonDir, "**", "*.json"), recursive=True
        )
        jsonNames = sorted(jsonNames)
        trainJsonNames, valJsonNames, testJsonNames = self._trainTestSplit(
            jsonNames, valSize, testSize
        )

        self._makeTrainValDir()

        dirs = ("train/", "val/", "test/")
        names = (trainJsonNames, valJsonNames, testJsonNames)
        for targetDir, jsonNames in zip(dirs, names):
            targetPart = targetDir.replace("/", "")
            logger.info("Converting %s set ...", targetPart)
            for jsonName in tqdm.tqdm(jsonNames):
                self.covertJsonToText(targetDir, jsonName)

        self._saveDatasetYaml()

    def covertJsonToText(self, targetDir, jsonName):
        with open(jsonName, encoding="utf-8") as file:
            jsonData = json.load(file)

        filename: str = uuid.UUID(int=random.Random().getrandbits(128)).hex
        imageName = f"{filename}.png"
        labelName = f"{filename}.txt"
        imgPath = saveYoloImage(
            jsonData, self._jsonDir, self._imageDirPath, targetDir, imageName
        )
        yoloObjList = self._getYoloObjectList(jsonData, imgPath)
        saveYoloLabel(yoloObjList, self._labelDirPath, targetDir, labelName)

    def convertOne(self, jsonName):
        jsonPath = os.path.join(self._jsonDir, jsonName)
        with open(jsonPath, encoding="utf-8") as file:
            jsonData = json.load(file)

        imageName = jsonName.replace(".json", ".png")
        labelName = jsonName.replace(".json", ".txt")
        imgPath = saveYoloImage(
            jsonData, self._jsonDir, self._imageDirPath, "", imageName
        )

        yoloObjList = self._getYoloObjectList(jsonData, imgPath)
        saveYoloLabel(yoloObjList, self._labelDirPath, "", labelName)

    def _getYoloObjectList(self, jsonData, imgPath):
        yoloObjList = []

        imgH, imgW, _ = cv2.imread(imgPath).shape
        for shape in jsonData["shapes"]:
            if shape["shape_type"] == "circle":
                yoloObj = self._getCircleShapeYoloObject(shape, imgH, imgW)
            else:
                yoloObj = self._getOtherShapeYoloObject(shape, imgH, imgW)

            if yoloObj:
                yoloObjList.append(yoloObj)

        return yoloObjList

    def _getCircleShapeYoloObject(self, shape, imgH, imgW):
        objCenterX, objCenterY = shape["points"][0]

        radius = math.sqrt(
            (objCenterX - shape["points"][1][0]) ** 2
            + (objCenterY - shape["points"][1][1]) ** 2
        )
        objW = 2 * radius
        objH = 2 * radius

        yoloCenterX = round(float(objCenterX / imgW), 6)
        yoloCenterY = round(float(objCenterY / imgH), 6)
        yoloW = round(float(objW / imgW), 6)
        yoloH = round(float(objH / imgH), 6)

        if shape["label"]:
            label = shape["label"]
            if label not in self._labelList:
                self._updateIdMap(label)
            labelId = self._labelIdMap[shape["label"]]

            return labelId, yoloCenterX, yoloCenterY, yoloW, yoloH

        return None

    def _getOtherShapeYoloObject(self, shape, imgH, imgW):
        pointList = shape["points"]
        points = np.zeros(2 * len(pointList))
        points[::2] = [float(point[0]) / imgW for point in pointList]
        points[1::2] = [float(point[1]) / imgH for point in pointList]

        if len(points) == 4:
            if self._outputFormat == "polygon":
                points = extendPointList(points)
            if self._outputFormat == "bbox":
                points = extendPointList(points, "bbox")

        if shape["label"]:
            label = shape["label"]
            if label not in self._labelList:
                self._updateIdMap(label)
            labelId = self._labelIdMap[shape["label"]]

            return labelId, points.tolist()

        return None

    def _saveDatasetYaml(self):
        yamlPath = os.path.join(self._jsonDir, "dataset/", "dataset.yaml")

        with open(yamlPath, "w+", encoding="utf-8") as yamlFile:
            trainDir = os.path.join(self._imageDirPath, "train/")
            valDir = os.path.join(self._imageDirPath, "val/")
            testDir = os.path.join(self._imageDirPath, "test/")

            namesStr = ""
            for label, _ in self._labelIdMap.items():
                namesStr += f'"{label}", '
            namesStr = namesStr.rstrip(", ")

            content = (
                f"train: {trainDir}\nval: {valDir}\ntest: {testDir}\n"
                f"nc: {len(self._labelIdMap)}\n"
                f"names: [{namesStr}]"
            )

            yamlFile.write(content)