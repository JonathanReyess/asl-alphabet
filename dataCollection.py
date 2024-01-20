import cv2
from cvzone.HandTrackingModule import HandDetector 
import numpy as np
import math
import time 
import os 

cap = cv2.VideoCapture(0) #our video capture object, 0 is  the id number of our webcam
detector = HandDetector(maxHands=2) #we will only be tracking one hand for our data collection

offset = 20
imgSize = 800


counter = 0
folder_base = "images"
current_folder = "A"

while True:
    success, img = cap.read() #img will be what our webcam video capture
    if not success:
        print("Failed to capture frame. Exiting...")
        break
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

        else:
            k = imgSize / w #we are stretching the height, this will be our constant
            calculatedHeight = math.ceil(k*h) #(if it's 3.5 it will always go to 4)
            imgResize = cv2.resize(imgCrop, (calculatedHeight, imgSize))
            imgResizeShape = imgResize.shape

            heightGap = math.ceil((imgSize - calculatedHeight)/2) #this is the gap we need to shift by to center the image

            imgWhite[:, heightGap:calculatedHeight+heightGap] = imgResize #starting point of height will be 0 and ending point will be height of imgcrop
    

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        

    cv2.imshow("Image", img) #show your webcam
    key = cv2.waitKey(1) #1 milisecond delay 
    if key == ord("s"):  # if we press the s key then we will save the image
        folder = os.path.join(folder_base, current_folder)
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(f'{folder}/Image_{time.time()}.png', imgWhite)
        print(f"Saved image in folder {current_folder}")
    elif key == ord("a"):  # if the user presses the a key, move to the next letter folder
        current_folder = chr((ord(current_folder) - ord('A') + 1) % 26 + ord('A'))
        print(f"Moved to the next folder: {current_folder}")
    elif key == 27:  # if the user presses the Esc key, exit the loop
        break



    #by centering our image and maintaining the same size, we are making it easier for our classifier 
        