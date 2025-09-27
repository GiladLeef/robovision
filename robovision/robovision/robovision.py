import math
import time
import cv2
import numpy as np
import onnxruntime

from robovision.utils import xywh2xyxy, nms, drawDetections, sigmoid

class Robovision:

    def __init__(self, path, confThres=0.7, iouThres=0.5, numMasks=32):
        self.confThreshold = confThres
        self.iouThreshold = iouThres
        self.numMasks = numMasks

        self.initializeModel(path)

    def __call__(self, image):
        return self.segmentObjects(image)

    def initializeModel(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        self.getInputDetails()
        self.getOutputDetails()

    def segmentObjects(self, image):
        inputTensor = self.prepareInput(image)

        outputs = self.inference(inputTensor)

        self.boxes, self.scores, self.classIds, maskPred = self.processBoxOutput(outputs[0])
        self.maskMaps = self.processMaskOutput(maskPred, outputs[1])

        return self.boxes, self.scores, self.classIds, self.maskMaps

    def prepareInput(self, image):
        self.imgHeight, self.imgWidth = image.shape[:2]

        inputImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputImg = cv2.resize(inputImg, (self.inputWidth, self.inputHeight))

        inputImg = inputImg / 255.0
        inputImg = inputImg.transpose(2, 0, 1)
        inputTensor = inputImg[np.newaxis, :, :, :].astype(np.float32)

        return inputTensor

    def inference(self, inputTensor):
        start = time.perf_counter()
        outputs = self.session.run(self.outputNames, {self.inputNames[0]: inputTensor})

        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def processBoxOutput(self, boxOutput):

        predictions = np.squeeze(boxOutput).T
        numClasses = boxOutput.shape[1] - self.numMasks - 4

        scores = np.max(predictions[:, 4:4+numClasses], axis=1)
        predictions = predictions[scores > self.confThreshold, :]
        scores = scores[scores > self.confThreshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        boxPredictions = predictions[..., :numClasses+4]
        maskPredictions = predictions[..., numClasses+4:]

        classIds = np.argmax(boxPredictions[:, 4:], axis=1)

        boxes = self.extractBoxes(boxPredictions)

        indices = nms(boxes, scores, self.iouThreshold)

        return boxes[indices], scores[indices], classIds[indices], maskPredictions[indices]

    def processMaskOutput(self, maskPredictions, maskOutput):

        if maskPredictions.shape[0] == 0:
            return []

        maskOutput = np.squeeze(maskOutput)

        numMask, maskHeight, maskWidth = maskOutput.shape
        masks = sigmoid(maskPredictions @ maskOutput.reshape((numMask, -1)))
        masks = masks.reshape((-1, maskHeight, maskWidth))

        scaleBoxes = self.rescaleBoxes(self.boxes,
                                   (self.imgHeight, self.imgWidth),
                                   (maskHeight, maskWidth))

        maskMaps = np.zeros((len(scaleBoxes), self.imgHeight, self.imgWidth))
        blurSize = (int(self.imgWidth / maskWidth), int(self.imgHeight / maskHeight))
        for i in range(len(scaleBoxes)):

            scaleX1 = int(math.floor(scaleBoxes[i][0]))
            scaleY1 = int(math.floor(scaleBoxes[i][1]))
            scaleX2 = int(math.ceil(scaleBoxes[i][2]))
            scaleY2 = int(math.ceil(scaleBoxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scaleCropMask = masks[i][scaleY1:scaleY2, scaleX1:scaleX2]
            cropMask = cv2.resize(scaleCropMask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            cropMask = cv2.blur(cropMask, blurSize)

            cropMask = (cropMask > 0.5).astype(np.uint8)
            maskMaps[i, y1:y2, x1:x2] = cropMask

        return maskMaps

    def extractBoxes(self, boxPredictions):
        boxes = boxPredictions[:, :4]

        boxes = self.rescaleBoxes(boxes,
                                   (self.inputHeight, self.inputWidth),
                                   (self.imgHeight, self.imgWidth))

        boxes = xywh2xyxy(boxes)

        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.imgWidth)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.imgHeight)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.imgWidth)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.imgHeight)

        return boxes

    def drawDetections(self, image, drawScores=True, maskAlpha=0.4):
        return drawDetections(image, self.boxes, self.scores,
                               self.classIds, maskAlpha)

    def drawMasks(self, image, drawScores=True, maskAlpha=0.5):
        return drawDetections(image, self.boxes, self.scores,
                               self.classIds, maskAlpha, maskMaps=self.maskMaps)

    def getInputDetails(self):
        modelInputs = self.session.get_inputs()
        self.inputNames = [modelInputs[i].name for i in range(len(modelInputs))]

        self.inputShape = modelInputs[0].shape
        self.inputHeight = self.inputShape[2]
        self.inputWidth = self.inputShape[3]

    def getOutputDetails(self):
        modelOutputs = self.session.get_outputs()
        self.outputNames = [modelOutputs[i].name for i in range(len(modelOutputs))]

    @staticmethod
    def rescaleBoxes(boxes, inputShape, imageShape):
        inputShape = np.array([inputShape[1], inputShape[0], inputShape[1], inputShape[0]])
        boxes = np.divide(boxes, inputShape, dtype=np.float32)
        boxes *= np.array([imageShape[1], imageShape[0], imageShape[1], imageShape[0]])

        return boxes