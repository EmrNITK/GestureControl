# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:59:07 2021

@author: Kritika
"""

import cv2
import numpy as np


def get_trackbars():
    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 640, 240)
    cv2.createTrackbar('HueMin', 'Trackbars', 0, 179, empty)
    cv2.createTrackbar('HueMax', 'Trackbars', 0, 179, empty)
    cv2.createTrackbar('SatMin', 'Trackbars', 0, 255, empty)
    cv2.createTrackbar('SatMax', 'Trackbars', 0, 255, empty)
    cv2.createTrackbar('ValMin', 'Trackbars', 0, 255, empty)
    cv2.createTrackbar('ValMax', 'Trackbars', 0, 255, empty)

def create_mask():
    while True:
        img = cv2.imread('') # read the current frame
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_min = cv2.getTrackbarPos('HueMin', 'Trackbars')
        hue_max = cv2.getTrackbarPos('HueMax', 'Trackbars')
        sat_min = cv2.getTrackbarPos('SatMin', 'Trackbars')
        sat_max = cv2.getTrackbarPos('SatMax', 'Trackbars')
        val_min = cv2.getTrackbarPos('ValMin', 'Trackbars')
        val_max = cv2.getTrackbarPos('ValMax', 'Trackbars')
        lower = np.array([hue_min, sat_min, val_min])
        upper = np.array([hue_max, sat_max, val_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        cv2.imshow('Mask', mask)



vid = cv2.VideoCapture(0);
while(1):
    _,frame = vid.read()
    
    cv2.imshow('video',frame)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
vid.release()

cv2.destroyAllWindows()    
