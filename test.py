import cv2
from cvzone.HandTrackingModule import HandDetector 
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0) #our video capture object, 0 is  the id number of our webcam
detector = HandDetector(maxHands=2) #we will only be tracking one hand for our data collection
classifier = Classifier("model/keras_model.h5", "model/labels.txt")
offset = 20
imgSize = 800

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
counter = 0
folder_base = "images"
current_folder = "A"

while True:
    success, img = cap.read() #img will be what our webcam video capture
    if not success:
        print("Failed to capture frame. Exiting...")
        break
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0] #we only have one hand
        x, y, w, h = hand['bbox'] #give us all the values from the hand 

        #we want all our images to be the same size for data collection so we will create a matrix with numpy
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 #unsigned integer of 8 bits (from 0 to 255)

        imgCrop =  img[y-offset:y+h+offset, x-offset:x+w+offset]  #our img is matrix, define starting height and width and ending h&w

        imgCropShape = imgCrop.shape

        aspectRatio = h / w #if value is above 1, then height is greater

        if aspectRatio > 1: #we want a fixed height and center it depending on width
            k = imgSize / h #we are stretching the height, this will be our constant
            calculatedWidth = math.ceil(k*w) #(if it's 3.5 it will always go to 4)
            imgResize = cv2.resize(imgCrop, (calculatedWidth, imgSize))
            imgResizeShape = imgResize.shape

            widthGap = math.ceil((imgSize - calculatedWidth)/2) #this is the gap we need to shift by to center the image

            imgWhite[:, widthGap:calculatedWidth+widthGap] = imgResize #starting point of height will be 0 and ending point will be height of imgcrop

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            #print(prediction, index)

       


        else:
            k = imgSize / w #we are stretching the height, this will be our constant
            calculatedHeight = math.ceil(k*h) #(if it's 3.5 it will always go to 4)
            imgResize = cv2.resize(imgCrop, (calculatedHeight, imgSize))
            imgResize = imgResize.transpose(1, 0, 2)
            imgResizeShape = imgResize.shape

            heightGap = math.ceil((imgSize - calculatedHeight)/2) #this is the gap we need to shift by to center the image
            print(imgWhite.shape, imgResize.shape)
            imgWhite[heightGap:calculatedHeight+heightGap, :] = imgResize #starting point of height will be 0 and ending point will be height of imgcrop
       
            prediction, index = classifier.getPrediction(imgWhite, draw=False)


        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
        

    cv2.imshow("Image", imgOutput) #show your webcam
    cv2.waitKey(1) #1 milisecond delay 

