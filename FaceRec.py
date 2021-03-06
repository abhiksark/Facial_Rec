#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 18:02:42 2017

@author: abhik
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os import listdir
from os.path import isfile,join
import numpy as np
import cv2
from skimage.feature import hog as HOG
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
import os
#Importing Models

from sklearn.svm import SVC,NuSVC

#Data Preparation

trainFolder = './data/' 

###############################################################################
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

Test = trainFolder +'Test/'
trainData = []
responseData = []
testData = []
NumberList = []

###############################################################################
ModiImages = [ f for f in listdir(Modi) if isfile(join(Modi,f)) ]
RagaImages = [ f for f in listdir(Raga) if isfile(join(Raga,f)) ]
KohliImages = [ f for f in listdir(Kohli) if isfile(join(Kohli,f)) ]
SalmanImages = [ f for f in listdir(Salman) if isfile(join(Salman,f)) ]
SainaImages = [ f for f in listdir(Saina) if isfile(join(Saina,f)) ]
SandbergImages = [ f for f in listdir(Sandberg) if isfile(join(Sandberg,f)) ]
GalImages = [ f for f in listdir(Gal) if isfile(join(Gal,f)) ]
MuskImages = [ f for f in listdir(Musk) if isfile(join(Musk,f)) ]
PriyankaImages = [ f for f in listdir(Priyanka) if isfile(join(Priyanka,f)) ]

###############################################################################

def ReadImages(ListName,FolderName,Label):
    #global NumberList
    global responseData
    global trainData
    #global hog
    #global cv2
    #global imutils
    #global winSize
    global testData
    #global os
    
   
    global feature 
   
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
    
def num2name(num):
    if num==0:
        name = 'Modi'
    elif num ==1:
        name = 'Raga'
    elif num ==2:
        name = 'Kohli'
    elif num==3:
        name = 'Salman'
    elif num ==4:
        name = 'Priyanka'
    elif num ==5:
        name = 'Musk'
    elif num==6:
        name = 'Gal'
    elif num==7:
        name = 'Sandberg'
    elif num==8:
        name = 'Saina'
    return name


ReadImages(ModiImages,Modi,0)
ReadImages(RagaImages,Raga,1)
ReadImages(KohliImages,Kohli,2)
ReadImages(SalmanImages,Salman,3)
ReadImages(PriyankaImages,Priyanka,4)
ReadImages(MuskImages,Musk,5)
ReadImages(GalImages,Gal,6)
ReadImages(SandbergImages,Sandberg,7)
ReadImages(SainaImages,Saina,8)



svm = NuSVC()
nu_options = np.arange(0.1,1)
kernel_options = ['linear','rbf']
param_grid= dict(kernel=kernel_options,nu = nu_options)
gridSVM = GridSearchCV(svm,param_grid,scoring = 'accuracy')
X = np.float32(trainData)
y = np.float32(responseData)
gridSVM.fit(X,y)
print(gridSVM.best_score_)
print(gridSVM.best_estimator_)
face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

TestImages = [ f for f in listdir(Test) if isfile(join(Test,f)) ]
for image in TestImages:
    img = cv2.imread(join(Test,image))
    imgray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    face = face_cascade.detectMultiScale(imgray)
    if len(face)>0:
            feature = HOG(cv2.resize(imgray[face[0][1]:face[0][1]+face[0][3],face[0][0]:face[0][0]+face[0][2]],
                                 (100,100)))
    pred = gridSVM.predict(feature)
    plt.figure()
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title(num2name(pred))
    
    

