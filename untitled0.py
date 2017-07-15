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
    noise =np.random.normal(0,1,size=(img.shape))
    imNoise=img+noise
    return imNoise

def rotate_image(img):
    rotImage= imutils.rotate(img,random.uniform(0,45))
    return rotImage 
#Data Preparation



trainFolder = './data/' 

Modi = trainFolder+'Modi/'
Raga= trainFolder+'Raga/'
Kohli= trainFolder+'Kohli/'
Salman =  trainFolder+'Salman/'

SalmanImages = [ f for f in listdir(Salman) if isfile(join(Salman,f)) ]

Test = trainFolder +'Test/'
trainData = []
responseData = []
testData = []
NumberList = []
ModiImages = [ f for f in listdir(Modi) if isfile(join(Modi,f)) ]
RagaImages = [ f for f in listdir(Raga) if isfile(join(Raga,f)) ]
KohliImages = [ f for f in listdir(Raga) if isfile(join(Kohli,f)) ]
FolderNames = [Modi,Raga,Kohli,Salman]
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
        
def ReadImages(ListName,FolderName,Label):   
    for image in ListName:
        face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = cv2.imread(join(FolderName,image))
        imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        face = face_cascade.detectMultiScale(imgray)

        if len(face)>0:
            feature = HOG(cv2.resize(imgray[face[0][1]:face[0][1]+face[0][3],face[0][0]:face[0][0]+face[0][2]],
                                 (100,100)))
            trainData.append(feature.T)
            responseData.append(Label)
