# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:59:07 2021

@author: Kritika
"""

import cv2
import numpy as np

def empty(a):
    pass

def create_trackbars():
    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 640, 240)
    cv2.createTrackbar('HueMin', 'Trackbars', 0, 179, empty)
    cv2.createTrackbar('HueMax', 'Trackbars', 0, 179, empty)
    cv2.createTrackbar('SatMin', 'Trackbars', 0, 255, empty)
    cv2.createTrackbar('SatMax', 'Trackbars', 0, 255, empty)
    cv2.createTrackbar('ValMin', 'Trackbars', 0, 255, empty)
    cv2.createTrackbar('ValMax', 'Trackbars', 0, 255, empty)

def create_mask(img):
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
    #cv2.imshow('Mask', mask)
    return mask

def threshold(mask):
    _,thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY) # if pixel intensity <= 127 then set it as 0 and pixel intensity > 127 set it as 255
    return thresh

def find_contours(thresh):
    contours,heirarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #give list of all essential boundary points
    return contours
    
def max_contour(contours):
    if len(contours) == 0:
        return []
    max_cntr = max(contours,key = cv2.contourArea)
    epsilon = 0.01*cv2.arcLength(max_cntr,True)  # maximum distance from contour to approximated contour. It is an accuracy parameter
    max_cntr = cv2.approxPolyDP(max_cntr,epsilon,True)
    return max_cntr

def centroid(contour):

    if len(contour) == 0: # if the array is empty return (-1,-1) 
        return (-1,-1)
    M=cv2.moments(contour) # gives a dictionary of all moment values calculated
    try:
        x = int(M['m10']/M['m00'])  # Centroid is given by the relations, 𝐶𝑥 =𝑀10/𝑀00 and 𝐶𝑦 =𝑀01/𝑀00
        y = int(M['m01']/M['m00'])
    except ZeroDivisionError:
        return (-1,-1) 
    return (x,y)

vid = cv2.VideoCapture(0);
create_trackbars()
while(1):
    _,frame = vid.read()
    frame = cv2.flip(frame,1) # resolving mirror image issues
    frame = frame[:300, 300:] # only considering frame from row 0-300 and col from 300-end so that main focus is on our hands
    frame = cv2.GaussianBlur(frame,(5,5),0) # to remove noise from frame

    mask = create_mask(frame)
    threshImg = threshold(mask)
    contours = find_contours(threshImg)
    frame = cv2.drawContours(frame,contours,-1,(255,0,0),2) # drawing all contours 
    max_cntr = max_contour(contours)  #finding maximum contour of the thresholded area
    (centroid_x,centroid_y) = centroid(max_cntr) #finding centroid of the maximum contour
    if(centroid_x,centroid_y) != (-1,-1):
        frame = cv2.circle(frame , (centroid_x,centroid_y) , 5 , (255,0,0) , -1) # drawing a circle on the identified centre of mass
    
    cv2.imshow('video',frame)
    cv2.imshow("mask",mask)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
vid.release()

cv2.destroyAllWindows()    
