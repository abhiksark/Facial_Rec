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
        self.responseData=[]
        self.trainFolder = './data/' 
        self.reverseName=dict()
        #self.reverseKey=0
    
    def num2name(self,num):
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
    
    def takeInput(self,name,reverseKey):
        self.reverseName[reverseKey]=name
        name = self.trainFolder+str(name)+'/'
        image = [ f for f in listdir(name) if isfile(join(name,f)) ]
        self.ReadImages(image,name,int(reverseKey))
    
    def ReadImages(self,ListName,FolderName,Label):   
        for image in ListName:
            img = cv2.imread(join(FolderName,image))
            try:
                imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                face = self.face_cascade.detectMultiScale(imgray) 
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
    
    def returnReverseName(self):
        return self.reverseName
