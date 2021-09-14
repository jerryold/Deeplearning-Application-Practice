---
tags: Artifical Intelligence Practice&Readme
---
# 人臉情緒辨識-MobileNetV2
## 需使用套件(freeze.yml,[haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/tree/master/data/haarcascades))
```
* freeze.yml:安裝提供的conda freeze.yml裡的套件即可開始運行
* haarcascade_frontalface_default.xml:本次使用人臉檢測使用OpenCV自帶的Haar特徵檢測,可取得人臉特徵,須將此xml檔放置資料夾才可執行

```
## MobileNetV2

* 本次主要訓練Kaggle所提供的FER 2013資料及,實作使用MobileNetV2原框架,並在透過用Transfer learning-Tuning,weight will start from last check point的概念帶入,在 output of global pooling layer後新增專門的classification layer
* MobileNetV2 基於 MobileNetV1 做改進，提出了一個創新的 layer module: the inverted residual with linear bottleneck，能夠在提升準確度的同時也提升了速度，是由 Google 於 2018 年發表的輕量化模型。
* Model Architecture
![](https://i.imgur.com/DXmw6G9.png)
![](https://i.imgur.com/86nFHjS.png)


* 常見比較(ResNet,MobileNetV1,MobileNetV2)
![](https://i.imgur.com/jvtA7r1.png)



## 檔案介紹
* emotiondetect.py:可透過輸入照片的方式判斷,該圖片中的人情緒資訊
* emotion_my_model.h5:預設訓練好的檔案即可輸入進行使用
