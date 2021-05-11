import cv2
import tensorflow as tf 
import random 
import numpy as np 
import matplotlib.pyplot as plt 
import os 

model = tf.keras.models.load_model('mymodel.h5')


# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
while True:
    # Read the frame
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        img = frame[y:y+h,x:x+w]
        img = cv2.resize(img,(150,150))
        img = img.reshape((1,150,150,3))
        pred = model.predict(img)
        if(pred[0]>0.5):
            cv2.putText(frame,"Please Wear Your Mask !!!", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=2)  
        else:
            cv2.putText(frame,"Thanks for wearing a mask :)", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=2)  
    
    cv2.imshow('frame', frame)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
