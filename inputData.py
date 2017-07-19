# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 13:58:35 2017

@author: HP
"""

import cv2
from os import listdir
from os.path import isfile,join
from skimage.feature import hog as HOG

class inputPictures():
    def __init__(self):
        self.face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.trainData=[]
        self.testData =[]
        self.testDict = dict()
        self.responseData=[]
        self.trainFolder = './data/' 
        self.reverseName=dict()
        #self.reverseKey=0
    def takeInput(self,name,reverseKey):
        self.reverseName[reverseKey]=name
        name = self.trainFolder+str(name)+'/'
        image = [ f for f in listdir(name) if isfile(join(name,f)) ]
        self.ReadImages(image,name,int(reverseKey))
    def inputTestData(self,name):
        name = self.trainFolder+str(name)+'/'
        images = [ f for f in listdir(name) if isfile(join(name,f))]
        for image in images:
            img = cv2.imread(join(name,image))
            try:
                imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                face = self.face_cascade.detectMultiScale(imgray,minSize = (50, 50),scaleFactor = 1.25) 
                if len(face)>0:
                    feature = HOG(cv2.resize(imgray[face[0][1]:face[0][1]+face[0][3],face[0][0]:face[0][0]+face[0][2]],
                                         (100,100)))
                    self.testData.append(feature.T)
                    self.testDict[image]=feature.T
            except Exception as e:
                print(e)
                pass
        
    def ReadImages(self,ListName,FolderName,Label):   
        for image in ListName:
            img = cv2.imread(join(FolderName,image))
            try:
                imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                face = self.face_cascade.detectMultiScale(imgray,minSize = (50, 50),scaleFactor = 1.25) 
                if len(face)>0:
                    feature = HOG(cv2.resize(imgray[face[0][1]:face[0][1]+face[0][3],face[0][0]:face[0][0]+face[0][2]],
                                         (100,100)))
                    self.trainData.append(feature.T)
                    self.responseData.append(Label)
            except Exception as e:
                print(e)
                pass
    def returnTrainData(self):
        return self.trainData,self.responseData
    def returnTestData(self):
        return self.testData,self.testDict
    def returnReverseName(self):
        return self.reverseName
