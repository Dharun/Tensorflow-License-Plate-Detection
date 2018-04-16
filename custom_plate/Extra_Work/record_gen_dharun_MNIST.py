# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:35:47 2018

@author: Vasantha kumar
""" 

from utils import MNIST_data_to_TF_record as MNIST
import os
path=os.path.join('MNIST_data')
MNIST.run(path)