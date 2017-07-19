#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:06:24 2017

@author: abhik
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
X=df.iloc[:,1:].values
y=df.iloc[:,0:1].values


imagesObj = inputPictures()
imagesObj.inputTestData('Test')
X_mainTest,y_dict=imagesObj.returnTestData()


########################################################################
#####################################################################
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()
#y=y[:,:-1] why this ?
############################################################################

predDict = {0:'Raga',1:'Kohli',2:'RanveerS',3:'Salman',4:'Musk'}


from sklearn.preprocessing import scale
X = scale( X, axis=0, with_mean=True, with_std=True, copy=True )
X_mainTest = scale( X_mainTest, axis=0, with_mean=True, with_std=True, copy=True )


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.decomposition import PCA
pca = PCA(n_components = 850)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_mainTest = pca.transform(X_mainTest)

explained_variance = pca.explained_variance_ratio_
new = np.sum(explained_variance)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 550, init = 'uniform', activation = 'relu', input_dim = 850))
classifier.add(Dense(output_dim = 350, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 1000, nb_epoch = 10)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_predSingle = np.argmax(y_pred,axis=1)
y_testSingle = np.argmax(y_test,axis=1)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_testSingle, y_predSingle)
print accuracy_score(y_testSingle, y_predSingle)


y_mainPred = classifier.predict(X_mainTest)
y_mainPred = np.argmax(y_mainPred,axis=1)


trainFolder = './data/' 
Test = trainFolder +'Test/'
TestImages = y_dict.keys()
face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for image in TestImages:
    img = cv2.imread(join(Test,image))
    imgray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    face = face_cascade.detectMultiScale(imgray,minSize = (50, 50),scaleFactor = 1.25)
    feature=y_dict[image]
    feature = scale( feature, axis=0, with_mean=True, with_std=True, copy=True )    
    new = pca.transform(feature)
    pred = classifier.predict(new.reshape(1, -1))
    pred = np.argmax(pred,axis=1)
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
        
        




