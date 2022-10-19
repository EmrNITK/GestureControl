# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:59:07 2021

@author: Kritika
"""

import cv2
import numpy as np

vid = cv2.VideoCapture(0);
while(1):
    _,frame = vid.read()
    
    cv2.imshow('video',frame)
    
    var = cv2.waitKey(1)
    
    if var == ord('q'):
        break
    
vid.release()

cv2.destroyAllWindows()    