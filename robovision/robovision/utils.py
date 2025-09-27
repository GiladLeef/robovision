import numpy as np
import cv2

classNames = ['note']

classWidth = [
    0.3556,
]
                      
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(classNames), 3))


def nms(boxes, scores, iouThreshold):
    sortedIndices = np.argsort(scores)[::-1]

    keepBoxes = []
    while sortedIndices.size > 0:
        boxId = sortedIndices[0]
        keepBoxes.append(boxId)

        ious = computeIou(boxes[boxId, :], boxes[sortedIndices[1:], :])

        keepIndices = np.where(ious < iouThreshold)[0]

        sortedIndices = sortedIndices[keepIndices + 1]

    return keepBoxes


def computeIou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    intersectionArea = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    boxArea = (box[2] - box[0]) * (box[3] - box[1])
    boxesArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    unionArea = boxArea + boxesArea - intersectionArea

    iou = intersectionArea / unionArea

    return iou


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def drawDetections(image, boxes, scores, classIds, maskAlpha=0.3, maskMaps=None):
    imgHeight, imgWidth = image.shape[:2]
    size = min([imgHeight, imgWidth]) * 0.0006
    textThickness = int(min([imgHeight, imgWidth]) * 0.001)

    maskImg = drawMasks(image, boxes, classIds, maskAlpha, maskMaps)

    for box, score, classId in zip(boxes, scores, classIds):
        color = colors[classId]

        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(maskImg, (x1, y1), (x2, y2), color, 2)

        label = classNames[classId]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=textThickness)
        th = int(th * 1.2)

        cv2.rectangle(maskImg, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)

        cv2.putText(maskImg, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), textThickness, cv2.LINE_AA)

    return maskImg


def drawMasks(image, boxes, classIds, maskAlpha=0.3, maskMaps=None):
    maskImg = image.copy()

    for i, (box, classId) in enumerate(zip(boxes, classIds)):
        color = colors[classId]

        x1, y1, x2, y2 = box.astype(int)

        if maskMaps is None:
            cv2.rectangle(maskImg, (x1, y1), (x2, y2), color, -1)
        else:
            cropMask = maskMaps[i][y1:y2, x1:x2, np.newaxis]
            cropMaskImg = maskImg[y1:y2, x1:x2]
            cropMaskImg = cropMaskImg * (1 - cropMask) + cropMask * color
            maskImg[y1:y2, x1:x2] = cropMaskImg

    return cv2.addWeighted(maskImg, maskAlpha, image, 1 - maskAlpha, 0)


def drawComparison(img1, img2, name1, name2, fontsize=2.6, textThickness=3):
    (tw, th), _ = cv2.getTextSize(text=name1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=textThickness)
    x1 = img1.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img1, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (0, 115, 255), -1)
    cv2.putText(img1, name1,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), textThickness)

    (tw, th), _ = cv2.getTextSize(text=name2, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=textThickness)
    x1 = img2.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img2, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (94, 23, 235), -1)

    cv2.putText(img2, name2,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), textThickness)

    combinedImg = cv2.hconcat([img1, img2])
    if combinedImg.shape[1] > 3840:
        combinedImg = cv2.resize(combinedImg, (3840, 2160))

    return combinedImg
