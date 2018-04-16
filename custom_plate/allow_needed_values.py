#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:27:59 2018
    Script filters out the unwanted values and prints
    values needed for License plate 
    [DESIGNED FOR TESSERACT]
@author: tensorflow-cuda
"""

def check_array(tex):
    side=tex
    az='ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    i=0
    yo=len(az)
    txt=''
    for i in range(0,yo):
        if side==az[i]:
            txt=az[i]
            break
    return txt   
def catch_rectify_plate_characters(text):
    tex = text
    out1=[]
    size=len(tex)
    for i in range(0,size):
      if tex[i]==check_array(tex[i]):
        out1.append(tex[i])
    yup=''.join(str(e) for e in out1)    
    return yup


