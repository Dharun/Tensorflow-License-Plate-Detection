# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:55:28 2018

@author: Vasantha kumar
"""

import cv2
import numpy as cv
#from cv2.cv import *
#import cv
cap=cv2.VideoCapture('video123.mp4')
width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
FPS=10
video=cv2.VideoWriter('bro_peace.avi',cv2.VideoWriter_fourcc(*'MJPG'),10,(640,360),True)
ret = True
while (ret):
    ret,img=cap.read()
    cv2.imshow('image',img) 
#    cv2.resize(img,(1280,960))
    video.write(img)
#    k=cv2.waitKey(10)&0xff 
#    if k==27:
#        break
#    cap.release()
#    video.release()
#    cv2.destroyAllWindows()
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
        cap.release()
        cv2.destroyAllWindows
        video.release