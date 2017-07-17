# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 00:43:50 2017

@author: Abhik
"""
from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
#import tensorflow as tf
import matplotlib.pyplot as plt
import imutils
import random
#Function
global trainFolder
trainFolder = './data/' 
global FolderName
FolderNames =[]
def add_noise(img):
    try:
        noise =np.random.normal(0,1,size=(img.shape))
        imNoise=img+noise
    except AttributeError:
        imNoise = img
    return imNoise

def rotate_image(img):
    try:
        rotImage= imutils.rotate(img,random.uniform(0,10))
    except AttributeError:
        rotImage = img 
    return rotImage 


def augmentImages(name):
    name = str(trainFolder) + str(name) +'/'
    FolderNames.append(name)
    #Images = [ f for f in listdir(name) if isfile(join(name,f)) ]
    
#Data Preparation

augmentImages('Musk')
augmentImages('Sandberg')
augmentImages('RanveerS')
augmentImages('Raga')
augmentImages('Modi')
augmentImages('Kohli')
augmentImages('Salman')
augmentImages('Obama')
augmentImages('Priyanka')

k=0
for folder in FolderNames:
    k=k+1
    for image in listdir(folder):
        img =cv2.imread(join(folder,image))
        noisedImg =add_noise(img)
        cv2.imwrite(str(folder)+'/Noised'+str(image),noisedImg)
        noisedImg2 =add_noise(img)
        cv2.imwrite(str(folder)+'/Noised2'+str(image),noisedImg2)
        noisedImg3 =add_noise(img)
        cv2.imwrite(str(folder)+'/Noised3'+str(image),noisedImg3)
        rotImage = rotate_image(img)
        cv2.imwrite(str(folder)+'/Rotated'+str(image),rotImage)
        rotImage2 = rotate_image(img)
        cv2.imwrite(str(folder)+'/Rotated2'+str(image),rotImage2)
        rotImage3 = rotate_image(img)
        cv2.imwrite(str(folder)+'/Rotated2'+str(image),rotImage3)




