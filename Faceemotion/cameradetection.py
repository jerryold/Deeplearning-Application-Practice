#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[ ]:


import cv2
from deepface import DeepFace
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(1)
cap.set(3,720) #hegiht
cap.set(4,720) #width
#check if the webcam is opened correctly
if not cap.isOpened():
    cap=cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot Open webcam")


while True:
    ret,frame=cap.read() #read one image from a video
    
    result=DeepFace.analyze(frame,actions=['emotion'])
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    faces=faceCascade.detectMultiScale(gray,1.1,4)

    #Draw the rectangle around the faces
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    font=cv2.FONT_HERSHEY_SIMPLEX
    


    #Use outText() method for
    #inserting text on video
    cv2.putText(frame,
                result['dominant_emotion'],
                (50,50),
                font,3,
                (0,0,255),
                2,
                cv2.LINE_4)
    cv2.imshow('Demo Video',frame)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

# 釋放攝影機
cap.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()


# In[ ]:




