---
tags: Artifical Intelligence Practice&Readme
---
# 人臉情緒辨識
## 需使用套件(requirments,[haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/tree/master/data/haarcascades))
```
* requirements.txt:安裝提供的requirments裡的套件即可開始運行
* haarcascade_frontalface_default.xml:本次使用人臉檢測使用OpenCV自帶的Haar特徵檢測,可取得人臉特徵,須將此xml檔放置資料夾才可執行

```
## Deepface
![](https://i.imgur.com/pycPav2.png)
* 本次使用Deepface作為我們所使用的library,Deepface is a lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for python. It is a hybrid face recognition framework wrapping state-of-the-art models: VGG-Face, Google FaceNet, OpenFace, Facebook DeepFace, DeepID, ArcFace and Dlib. Those models already reached and passed the human level accuracy. The library is mainly based on TensorFlow and Keras.
 
* Deepface is a hybrid face recognition package. It currently wraps many state-of-the-art face recognition models: VGG-Face , Google FaceNet, OpenFace, Facebook DeepFace, DeepID, ArcFace and Dlib. The default configuration uses VGG-Face model.

![](https://i.imgur.com/3mzEGT6.png)
    ** FaceNet, VGG-Face, ArcFace and Dlib overperforms than OpenFace, DeepFace and DeepID based on experiments. Supportively, FaceNet /w 512d got 99.65%; FaceNet /w 128d got 99.2%; ArcFace got 99.41%; Dlib got 99.38%; VGG-Face got 98.78%; DeepID got 97.05; OpenFace got 93.80% accuracy scores on LFW data set whereas human beings could have just 97.53%.




## 檔案介紹
* emotiondetect.py:可透過輸入照片的方式判斷,該圖片中的人情緒,年齡,膚色......等等相關資訊
* cameradetection.py:可即時偵測攝影機當中的人情緒,若有新增其他偵測內容請自行去程式碼中更改