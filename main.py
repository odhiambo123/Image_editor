import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

image_path = 'images'
images = os.listdir(image_path)

image_index = 0
bg_image = cv2.imread(image_path+ '/'+ images[image_index])

#1 Initialize selfie segmentation object

#initialize mediakpipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

#create a video capture object to access the webcam
cap = cv2.VideoCapture(0) #CAPTURE object that reads frames
while cap.isOpened(): # check if the capture object is available or not
    _, frame = cap.read() #read each frame from the webcam

    #flip the frame to horizontal direction
    frame = cv2.flip(frame,1)
    height, width, channel1 = frame.shape

    # create the segmented mask
    RGB = cv2.cvtColor(frame, cv2.COLOR_BAYER_GR2RGB)

def copy_image():
    link = input("Enter Image url:")
    new_file_name = "image1.jpg"
    urllib.request.urlretrieve(link, new_file_name)

    k = cv2.imread(new_file_name)
    #convert to graky
    gray = cv2.cvtColor(k, cv2.COLOR_BGR2GRAY)

    #threashold input image as mask
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

    #NEGATE MASK
    mask = 255 - mask
    #apply morphology to remove isolated extraneaous noise
    #use border constant of black since forground touches the edge
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #anti-alias the mask --blur the stretch
    #blur alpha channel

    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2,borderType=cv2.BORDER_DEFAULT)
    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2 * (mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = k.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask

    # save resulting masked image
    cv2.imwrite('img_no_bckgrnd.png', result)

    # display result, though it won't show transparency
    cv2.imshow("INPUT", k)
    cv2.imshow("GRAY", gray)
    cv2.imshow("MASK", mask)
    cv2.imshow("RESULT", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    copy_image()


