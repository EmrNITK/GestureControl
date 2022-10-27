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

leftEyeLmIndex = [263, 249, 390, 373, 374, 380, 381, 382,
                  362]  # indexs of left eye landmarks
rightEyeLmIndex = [33, 7, 163, 144, 145, 153, 154, 155,
                   133]  # indexs of right eye landmarks


def appendAndDrawEyesLandmarks(lmIndex, faceLandmarks, img, eyesLandmarks,
                               draw):
    """
    Args:
        lmIndex -> list of landmark indexs \n
        faceLandmarks -> mediapipe faceLandmarks \n
        img -> numpy image \n
        eyesLandmarks -> list to append coordinates \n
        draw -> bool to draw on ROI
    """
    imgHeight, imgWidth, imgChannel = img.shape  # extract image Height, and width
    for index in lmIndex:
        landmark = faceLandmarks.landmark[index]
        # convert normalized x, and y values to original values
        x_coord, y_coord = int(landmark.x * imgWidth), int(landmark.y *
                                                           imgHeight)
        # append coordinates
        eyesLandmarks.append([id, x_coord, y_coord])
        if draw == True:
            cv2.circle(img, (x_coord, y_coord), 1, (255, 0, 0), 2)


def detectEyes(img, show=False):
    """
    Args:
        img -> numpy image \n
        show -> bool to draw on ROI
    returns:
        number of ROIs detected
    """

    # convert BGR image to RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # process image
    results = faceMesh.process(imgRGB)

    eyesLandmarks = []

    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:
            # append and draw eyes landmarks
            appendAndDrawEyesLandmarks(leftEyeLmIndex, faceLandmarks, img,
                                       eyesLandmarks, show)
            appendAndDrawEyesLandmarks(rightEyeLmIndex, faceLandmarks, img,
                                       eyesLandmarks, show)

    total_landmarks = len(leftEyeLmIndex) + len(rightEyeLmIndex)
    eyesLandmarksLength = len(eyesLandmarks)

    # for debugging purpose
    if show:
        cv2.imshow('img', img)

    if eyesLandmarksLength < total_landmarks / 2:
        return 0
    elif eyesLandmarksLength >= total_landmarks / 2 and eyesLandmarksLength < total_landmarks:
        return 1
    elif eyesLandmarksLength == total_landmarks:
        return 2
