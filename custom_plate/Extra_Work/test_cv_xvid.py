# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 15:43:34 2018

@author: Dharun
"""

import numpy as np
import cv2
import os
path = os.path.join('frame_cv')
cap = cv2.VideoCapture(0)
count = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
#        frame = cv2.flip(frame,0)
        count += 1
        filename = os.path.join(path,'frame{}.jpg'.format(count))
#        cv2.imwrite("frame%d.jpg" % count, frame)
        cv2.imwrite(filename, frame)
        # write the flipped frame
#        out.write(frame)
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
#out.release()
cv2.destroyAllWindows()