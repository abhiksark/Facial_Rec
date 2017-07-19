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
#df = pd.read_csv("Faces.csv")
#Y = df[['Face']]
#print(Y)
#
#X=df.iloc[:,1:].values
#y=df.iloc[:,0:1].values
imagesObj = inputPictures()
#imagesObj.takeInput('Modi',0)
imagesObj.takeInput('Raga',0)
#imagesObj.takeInput('Gal',2)
imagesObj.takeInput('Kohli',1)
imagesObj.takeInput('RanveerS',2)
imagesObj.takeInput('Salman',3)
imagesObj.takeInput('Musk',4)
#imagesObj.takeInput('Sandberg',7)
#imagesObj.takeInput('Obama',6)
#imagesObj.takeInput('Priyanka',7)
#predDict={0:'Modi',1:'Raga',2:'Kohli',3:'RanveerS',4:'Salman',5:'Musk',6:'Obama',7:'Priyanka'}
#imagesObj.takeInput('Saina',8)
predDict = imagesObj.returnReverseName()
X,y=imagesObj.returnTrainData()
#
#y=np.array(y)
#y=np.resize(y,(4257,1))
#newX=np.asarray(X)
#new=np.concatenate((y,newX), axis=1)
#df = pd.DataFrame(new)
#df.to_csv("Faces.csv",index=False)

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
    face = face_cascade.detectMultiScale(imgray,minSize = (50, 50),scaleFactor = 1.25)
    if len(face)>0:
            feature = HOG(cv2.resize(imgray[face[0][1]:face[0][1]+face[0][3],face[0][0]:face[0][0]+face[0][2]],
                                 (100,100)))
    feature=scale( feature, axis=0, with_mean=True, with_std=True, copy=True )
    pred = svm.predict(feature.reshape(1, -1))
    plt.figure()
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title(predDict[int(pred)])
    try:
            image = cv2.imread(join(Test,image))
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
        
        


