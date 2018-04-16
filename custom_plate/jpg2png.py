#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 15:53:30 2018
import os
os.remove(file) for file in os.listdir('path/to/directory') if file.endswith('.png')
@author: tensorflow-cuda
"""
#import os
#from PIL import Image
#i=1
#
#
#os.remove(file) for file in os.listdir() if file.endswith('.jpeg')
#
#
#
#
#for i in range(1,162):
#   im = Image.open('image{}.jpg'.format(i))
#   im.save('image{}.png'.format(i))




from glob import glob                                                           
import cv2 
jpgs = glob('./*.JPG')

for j in jpgs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'png', img)