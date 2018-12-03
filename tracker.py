# organize imports
import cv2
import imutils
import numpy as np
import time
from skimage.measure import compare_ssim
# global variables
bg = None

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
    (score, diff) = compare_ssim(image, bg.astype("uint8"), full=True)
    diff = (diff * 255).astype("uint8")
    cv2.imwrite("a.png",diff)
    return diff

if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left =1, 250, 400, 650


    # initialize num of frames
    num_frames = 0
    image_num = 1

    start_recording = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

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
            print(num_frames)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                difference = hand

        	
                if start_recording:
		    
		    resized = cv2.resize(difference, (300, 300)) 
                    # Mention the directory in which you wanna store the images followed by the image name
                    cv2.imwrite("Dataset/paper/paper_" + str(image_num) + '.png', resized)
                    image_num += 1
                cv2.imshow("difference", difference)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q") or image_num > 400:
            break
        
        if keypress == ord("s"):
            start_recording = True


# free up memory
camera.release()
cv2.destroyAllWindows()
