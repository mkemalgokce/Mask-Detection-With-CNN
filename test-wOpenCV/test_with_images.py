import cv2
import tensorflow as tf 
import random 
import numpy as np 
import matplotlib.pyplot as plt 
import os 

model = tf.keras.models.load_model('mymodel.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

frame = cv2.imread('masked.jpg')
img = frame
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

img = cv2.resize(img,(150,150))
img = img.reshape((1,150,150,3))
pred = model.predict(img)
if(pred[0]>0.5):
    cv2.putText(frame,"Please Wear Your Mask !!!", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0),thickness=2)  
else:
    cv2.putText(frame,"Thanks for wearing a mask :)", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),thickness=2)  

cv2.imshow('IMAGE', frame)
    # Stop if escape key is pressed

cv2.waitKey(0)
cv2.destroyAllWindows()