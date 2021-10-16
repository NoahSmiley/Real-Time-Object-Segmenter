import cv2
import numpy as np

# for webcam connection
cap = cv2.VideoCapture(0)

whT = 320

confidenceThreshold = 0.3
nmsThreshold = 0.3

# Imports class name files, and extracts and
# seperates each name to a new string element
# within the classNames array.
# This is simply just an array of all possible
# Detectable objects
classesFile = 'coco.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    ht, wt, ct = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                w, h = int(detection[2]*wt), int(detection[3]*ht)
                x, y = int((detection[0]*wt) - w/2), int((detection[1]*ht)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classID)
                confs.append(float(confidence))

    print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confidenceThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (251, 218, 97), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (251, 218, 97), 2)

cv2.startWindowThread()

while True:
    # gives us our image, and lets us know if it was  successfully retrieved
    success, img = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    # k = cv2.waitKey(0)

    # if k == 27:         # wait for ESC key to exit
    #     cv2.destroyAllWindows()

    # Takes our image and renders it and compiles it into binary
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()

    # List comprehension getting the layerName from the indexes
    # Created within our blob's layers 1-3

    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    findObjects(outputs, img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)

cv2.destroyAllWindows()