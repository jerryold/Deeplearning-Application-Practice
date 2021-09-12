#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2


# In[3]:


pip install deepface


# In[4]:


from deepface import DeepFace


# In[5]:


img=cv2.imread('happyboy.jpg')


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


plt.imshow(img) #BGR


# In[8]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[9]:


prediction=DeepFace.analyze(img)


# In[10]:


prediction


# In[11]:


type(prediction)
prediction['dominant_emotion']


# ### we are trying to draw a rectangle across the face

# In[12]:


faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


# In[13]:


gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
faces=faceCascade.detectMultiScale(gray,1.1,4)

    #Draw the rectangle around the faces
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


# In[14]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[15]:


font=cv2.FONT_HERSHEY_SIMPLEX

    #Use outText() method for
    #inserting text on video
cv2.putText(img,
               prediction['dominant_emotion'],
              (50,50),
              font,1,
              (0,0,255),
              2,
              cv2.LINE_4);
                
               


# In[16]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[17]:


img2=cv2.imread('fearman.jpg')
plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))


# In[18]:


prediction2=DeepFace.analyze(img2)


# In[19]:


prediction2

