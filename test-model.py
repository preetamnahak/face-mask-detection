# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 02:59:17 2020

@author: pradyumna
"""
import os
from keras.models import load_model
import cv2
import numpy as np

source=cv2.VideoCapture(0)
print(source.isOpened())
model = load_model('model-010.model')

face_clsfr=cv2.CascadeClassifier(os.getcwd() + '/dataset/haarcascade_frontalface_default.xml')



labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

while(1):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for x,y,w,h in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
        if result[0][0] > result[0][1]:
            val = round(result[0][0]*100, 2)
        else:
            val = round(result[0][1]*100, 2)
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-30),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label] + " " + str(val) + "%", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255),2)
        
        
    cv2.imshow('Face Mask Detector',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()
