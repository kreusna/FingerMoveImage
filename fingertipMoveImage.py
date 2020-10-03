import cv2
import numpy as np
import glob
import random

capture = cv2.VideoCapture(0)
# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3-tiny.cfg")

getImage = cv2.imread('baby.jpeg')
smallImage = cv2.resize( getImage, (75, 75))

smallImageoldx = 400
smallImageoldy = 350

xmiddleImage = 20
ymiddleImage = 20

xlastImage = 75 - xmiddleImage
ylastImage = 75 - ymiddleImage

smallImageoldwidth = smallImageoldx + xmiddleImage + xlastImage
smallImageoldhigh = smallImageoldy + ymiddleImage + ylastImage


layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

font = cv2.FONT_HERSHEY_PLAIN

def calculateDistance(topLists):
    result = [0,0,0]
    if len(topLists) > 1:
        top1x = topLists[0][0]
        top2x = topLists[1][0]
        distance = top1x - top2x
        if top2x > top1x:
            distance = top2x - top1x

        distanceX = int(top1x/2) + int(top2x/2)
        distanceY = int(topLists[0][1]/2) + int(topLists[1][1]/2)  
        result = [distance,distanceX, distanceY]
    return result

isTouch = 0
while True:
    _, frame = capture.read()
    touch = 0
    img = cv2.resize(frame, None, fx=1, fy=1)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    newTopdata = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                newTopdata.append([center_x, center_y])
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
    
    getDistanceFromTopFinger = calculateDistance(newTopdata)
    if (getDistanceFromTopFinger[0] < 30):
        touch = 1

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.7)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = (255,0,0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    
    if touch ==1:
        if getDistanceFromTopFinger[1] in range(smallImageoldx, smallImageoldwidth  ) and getDistanceFromTopFinger[2] in range(smallImageoldy, smallImageoldhigh ):
            if isTouch ==0:
                xmiddleImage = getDistanceFromTopFinger[1] - smallImageoldx
                ymiddleImage = getDistanceFromTopFinger[2] - smallImageoldy
                isTouch = 1

            smallImageoldx = getDistanceFromTopFinger[1] - xmiddleImage
            smallImageoldy = getDistanceFromTopFinger[2] - ymiddleImage

            xlastImage = 75 - xmiddleImage
            ylastImage = 75 - ymiddleImage    

            smallImageoldwidth = smallImageoldx + xmiddleImage + xlastImage
            smallImageoldhigh = smallImageoldy + ymiddleImage + ylastImage
            
    else:
        isTouch = 0
    img[smallImageoldy:smallImageoldhigh, smallImageoldx:smallImageoldwidth] = smallImage



    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()