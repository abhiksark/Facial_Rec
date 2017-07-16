# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 00:43:50 2017

@author: Abhik
"""
from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import imutils
import random
#Function

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
#Data Preparation



trainFolder = './data/' 

Modi = trainFolder+'Modi/'
Raga= trainFolder+'Raga/'
Kohli= trainFolder+'Kohli/'
Salman =  trainFolder+'Salman/'
Gal = trainFolder +'Gal/'
Musk =  trainFolder+'Musk/'
Sandberg =  trainFolder+'Sandberg/'
Salman =  trainFolder+'Salman/'
Saina =  trainFolder+'Saina/'
Priyanka =trainFolder+'Priyanka/'


ModiImages = [ f for f in listdir(Modi) if isfile(join(Modi,f)) ]
RagaImages = [ f for f in listdir(Raga) if isfile(join(Raga,f)) ]
KohliImages = [ f for f in listdir(Kohli) if isfile(join(Kohli,f)) ]
SalmanImages = [ f for f in listdir(Salman) if isfile(join(Salman,f)) ]
SainaImages = [ f for f in listdir(Saina) if isfile(join(Saina,f)) ]
SandbergImages = [ f for f in listdir(Sandberg) if isfile(join(Sandberg,f)) ]
GalImages = [ f for f in listdir(Gal) if isfile(join(Gal,f)) ]
MuskImages = [ f for f in listdir(Musk) if isfile(join(Musk,f)) ]
PriyankaImages = [ f for f in listdir(Priyanka) if isfile(join(Priyanka,f)) ]


Test = trainFolder +'Test/'
trainData = []
responseData = []
testData = []
NumberList = []

FolderNames = [Modi,Raga,Kohli,Salman,Musk,Sandberg,Saina,Priyanka,Gal]
k=0

for folder in FolderNames:
    k=k+1
    for image in listdir(folder):
        img =cv2.imread(join(folder,image))
        noisedImg =add_noise(img)
        cv2.imwrite(str(folder)+'/Noised'+str(image),noisedImg)
        noisedImg2 =add_noise(img)
        cv2.imwrite(str(folder)+'/Noised2'+str(image),noisedImg2)
        rotImage = rotate_image(img)
        cv2.imwrite(str(folder)+'/Rotated'+str(image),rotImage)
        rotImage2 = rotate_image(img)
        cv2.imwrite(str(folder)+'/Rotated2'+str(image),rotImage2)

