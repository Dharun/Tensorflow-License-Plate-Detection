#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:29:57 2018

@author: tensorflow-cuda
"""
import cv2
import os

def yo_make_the_conversion(img_data, count):
    img = img_data
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    path_png='png_tesseract'
    count += 1
    filename = os.path.join(path_png,'image{}.png'.format(count))
#    yo = tester.process_image_for_ocr(file)
    cv2.imwrite(filename, gray)
    return filename
