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
        face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = cv2.imread(join(FolderName,image))
        imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        face = face_cascade.detectMultiScale(imgray,minSize = (50, 50)) #,scaleFactor = 1.15, minNeighbors = 5,
        print(face)
        if len(face)>0:
            x, y, w, h =  face[0]
            cv2.rectangle(imgray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            pass
        cv2.imshow("Objects found", imgray)
        cv2.waitKey(100)
        sleep(0.5)
        cv2.destroyAllWindows()

ReadImages(ModiImages,Modi)



import sys, cv2

# Refactored https://realpython.com/blog/python/face-recognition-with-python/

def detections_draw(image, detections):
  for (x, y, w, h) in detections:
    print "({0}, {1}, {2}, {3})".format(x, y, w, h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

def main(argv = None):
  if argv is None:
    argv = sys.argv

  cascade_path = sys.argv[1]
  image_path = sys.argv[2]
  result_path = sys.argv[3] if len(sys.argv) > 3 else None

  cascade = cv2.CascadeClassifier(cascade_path)
  image = cv2.imread(image_path)
  if image is None:
    print "ERROR: Image did not load."
    return 2

  detections = cascade_detect(cascade, image)
  detections_draw(image, detections)

  print "Found {0} objects!".format(len(detections))
  if result_path is None:
    cv2.imshow("Objects found", image)
    cv2.waitKey(0)
  else:
    cv2.imwrite(result_path, image)