# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 14:54:09 2017

@author: HP
"""


from inputData import inputPictures
#Importing Models
import numpy as np
import cv2
from skimage.feature import hog as HOG
import matplotlib.pyplot as plt
from sklearn.svm import SVC,NuSVC
from os import listdir
from os.path import isfile,join
#from sklearn.grid_search import GridSearchCV
import pandas as pd
df = pd.read_csv("Faces.csv")
Y = df[['Face']]
print(Y)

X=df.iloc[:,1:].values
y=df.iloc[:,0:1].values
#df = pd.DataFrame([X,y])
#df.to_csv("faces.csv")
#imagesObj = inputPictures()
#imagesObj.takeInput('Modi',0)
#imagesObj.takeInput('Raga',1)
##imagesObj.takeInput('Gal',2)
#imagesObj.takeInput('Kohli',2)
#imagesObj.takeInput('RanveerS',3)
#imagesObj.takeInput('Salman',4)
#imagesObj.takeInput('Musk',5)
##imagesObj.takeInput('Sandberg',7)
#imagesObj.takeInput('Obama',6)
#imagesObj.takeInput('Priyanka',7)
predDict={0:'Modi',1:'Raga',2:'Kohli',3:'RanveerS',4:'Salman',5:'Musk',6:'Obama',7:'Priyanka'}
#imagesObj.takeInput('Saina',8)
#predDict = imagesObj.returnReverseName()
#X,y=imagesObj.returnTrainData()

#y=np.array(y)
#y=np.resize(y,(1326,1))
#newX=np.asarray(X)
#new=np.concatenate((y,newX), axis=1)
#df = pd.DataFrame(new)
#df.to_csv("Faces.csv")

from sklearn.preprocessing import scale
X = scale( X, axis=0, with_mean=True, with_std=True, copy=True )

svm = NuSVC(decision_function_shape = 'ovr',nu=0.1)
#nu_options = np.arange(0.1,1)
#kernel_options = ['linear','rbf']
#param_grid= dict(kernel=kernel_options,nu = nu_options)
#gridSVM = GridSearchCV(svm,param_grid,scoring = 'accuracy')
svm.fit(X,y)
#print(gridSVM.best_score_)
#print(gridSVM.best_estimator_)
face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


trainFolder = './data/' 
Test = trainFolder +'Test/'
TestImages = [ f for f in listdir(Test) if isfile(join(Test,f)) ]
for image in TestImages:
    img = cv2.imread(join(Test,image))
    imgray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    face = face_cascade.detectMultiScale(imgray)
    if len(face)>0:
            feature = HOG(cv2.resize(imgray[face[0][1]:face[0][1]+face[0][3],face[0][0]:face[0][0]+face[0][2]],
                                 (100,100)))
    feature=scale( feature, axis=0, with_mean=True, with_std=True, copy=True )
    pred = svm.predict(feature.reshape(1, -1))
    plt.figure()
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title(predDict[int(pred)])

