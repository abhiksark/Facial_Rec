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
from time import sleep
from skimage import data, color, exposure

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

global faceCount
faceCount =0

def ReadImages(ListName,FolderName):   
    for image in ListName:
#        face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#        img = cv2.imread(join(FolderName,image))
#        imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#        face = face_cascade.detectMultiScale(imgray,minSize = (100, 100),scaleFactor = 1.25) #,scaleFactor = 1.15, minNeighbors = 5,
#        print(face)
#        if len(face)>0:
#            x, y, w, h =  face[0]
#            cv2.rectangle(imgray, (x, y), (x + w, y + h), (0, 255, 0), 2)
#        else:
#            pass
#        cv2.imshow("Objects found", imgray)
#        cv2.waitKey(100)
#        sleep(0.3)
#        cv2.destroyAllWindows()
#        cv2.waitKey(1)
#        cv2.waitKey(1)
#        cv2.waitKey(1)
#        cv2.waitKey(1) #bug in openCV linux
        try:
            image = cv2.imread(join(FolderName,image))
            imgray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            face = face_cascade.detectMultiScale(imgray,minSize = (50, 50),scaleFactor = 1.25) 
            if len(face)>0:
                feature,hog_image= HOG(cv2.resize(imgray[face[0][1]:face[0][1]+face[0][3],face[0][0]:face[0][0]+face[0][2]],(100,100)),cells_per_block=(1, 1), visualise=True)                        
                cropped_image=imgray[face[0][1]:face[0][1]+face[0][3],face[0][0]:face[0][0]+face[0][2]]
                fig, (ax1, ax2) = plt.subplots(1, 2)
                
                ax1.axis('off')
                ax1.imshow(cropped_image, cmap=plt.cm.gray)
                ax1.set_title('Input image')
                ax1.set_adjustable('box-forced')
                
                # Rescale histogram for better display
                #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
                
                ax2.axis('off')
                ax2.imshow(hog_image, cmap=plt.cm.gray)
                ax2.set_title('Histogram of Oriented Gradients')
                ax1.set_adjustable('box-forced')
                plt.show()
            else:
                continue
        except Exception as e:
            print(e)
            pass

ReadImages(SalmanImages,Salman)

