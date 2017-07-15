# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 01:38:49 2017

@author: HP
"""

from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
from skimage.feature import hog as HOG
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
import os

trainFolder = './data/' 

Modi = trainFolder+'Modi/'
Raga= trainFolder+'Raga/'
Kohli= trainFolder+'Kohli/'
Salman = trainFolder + 'Salman/'
Test = trainFolder +'Test/'
trainData = []
responseData = []
testData = []
NumberList = []
ModiImages = [ f for f in listdir(Modi) if isfile(join(Modi,f)) ]
RagaImages = [ f for f in listdir(Raga) if isfile(join(Raga,f)) ]
KohliImages = [ f for f in listdir(Kohli) if isfile(join(Kohli,f)) ]
SalmanImages = [ f for f in listdir(Salman) if isfile(join(Salman,f)) ]


def ReadImages(ListName,FolderName):   
    for image in ListName:
        face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = cv2.imread(join(FolderName,image))
        imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        face = face_cascade.detectMultiScale(imgray)

        if len(face)>0:
            plt.imshow(cv2.resize(imgray[face[0][1]:face[0][1]+face[0][3],face[0][0]:face[0][0]+face[0][2]]))           
            
ReadImages(ModiImages,Modi)