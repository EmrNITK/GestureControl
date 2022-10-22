"""
author: @anugoyal998
"""

"""
To install opencv and mediapipe 
Run:
pip install opencv-python
pip install mediapipe
"""

import cv2
import mediapipe as mp

# init mediapipe faceMesh
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

leftEyeLmIndex = [263, 249, 390, 373, 374, 380, 381, 382, 362]
rightEyeLmIndex = [33, 7, 163, 144, 145, 153, 154, 155, 133]

def appendAndDrawEyesLandmarks(lmIndex,faceLandmarks,img,eyesLandmarks,draw):
    imgHeight, imgWidth, imgChannel = img.shape # extract image Height, and width
    for index in lmIndex:
        landmark = faceLandmarks.landmark[index]
        # convert normalized x, and y values to original values
        x_coord, y_coord = int(landmark.x * imgWidth), int(landmark.y * imgHeight)
        # append coordinates
        eyesLandmarks.append([id,x_coord,y_coord])
        if draw == True:
            cv2.circle(img,(x_coord,y_coord),1,(255,0,0),2)

def detectEyes(img,show=False):
    # convert BGR image to RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # process image
    results = faceMesh.process(img)
    
    eyesLandmarks = []
    
    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:
            # append and draw eyes landmarks 
            appendAndDrawEyesLandmarks(leftEyeLmIndex,faceLandmarks,img,eyesLandmarks,show)
            appendAndDrawEyesLandmarks(rightEyeLmIndex,faceLandmarks,img,eyesLandmarks,show)
    
    size = len(leftEyeLmIndex) + len(rightEyeLmIndex)
    eyesLandmarksLength = len(eyesLandmarks)
    
    cv2.imshow('img',img)
    
    if eyesLandmarksLength < size/2:
        return 0
    elif eyesLandmarksLength >= size/2 and eyesLandmarksLength < size:
        return 1
    elif eyesLandmarksLength >= size:
        return 2