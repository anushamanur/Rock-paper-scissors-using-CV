import tensorflow as tf
import tflearn
import numpy as np
from PIL import Image
import cv2
import imutils
import random
from skimage import feature
from sklearn.externals import joblib
from skimage.measure import compare_ssim
import time

# global variables
bg = None

def resizeImage(imageName):
    img = Image.open(imageName)
    img = img.resize((300,300))
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame

    cv2.imwrite("bg.png", bg)

    (score, diff) = compare_ssim(image, bg.astype("uint8"), full=True)
    diff = (diff * 255).astype("uint8")
    cv2.imwrite("a.png",diff)

    return diff


def getPredictedClass():
    # Predict
    image = cv2.imread('Temp.png',0)
    X=[]
    H = feature.hog(image, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    X.append(H)
    X = np.array(X)
    prediction = model.predict(X)
    print prediction
    return prediction


def showStatistics(predictedClass):

    textImage = np.zeros((300,300,3), np.uint8)
    className = ""
    

    if predictedClass == 0:
        className = "Paper"
    elif predictedClass == 1:
        className = "Rock"
    elif predictedClass == 2:
	className="Scissors"

    cv2.putText(textImage,"User : " + className, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)
    cv2.imshow("Statistics", textImage)



# load the model from disk
model = joblib.load('svm_model_2.sav')

if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 1, 250, 400, 650

    # initialize num of frames
    num_frames = 0
    start_recording = False
  

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width = 700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
	    # observe the keypress by the user
          
            
            if True:
                hand = segment(gray)

                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                
                    difference = hand

                    # draw the segmented region and display the frame
               
                    if start_recording:
                        cv2.imwrite('Temp.png', difference)
                        resizeImage('Temp.png')
               	    
		        predictedClass = getPredictedClass()
		        showStatistics(predictedClass)
			
                    
                    cv2.imshow("diff", difference)
		
        

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        
        if keypress == ord("s"):
            start_recording = True
	   






