---
tags: Artifical Intelligence Practice&Readme
---
# 人臉辨識
## 步驟流程
1. Face Detection
1. Face Align
1. Feature extraction
1. Create Database
1. Face Recognition
## Face Detection
* [MTCNN](https://pypi.org/project/mtcnn/)
* [Retina face](https://pypi.org/project/retinaface/)
```
若無法引入Retinaface請使用以下方式
conda config --add channels conda-forge
conda install shapely
```
`pip install retina-face`
* 偵測人臉~輸出會有預測框左上角跟右下角、兩個眼睛、鼻子、嘴巴兩邊的座標值
## Face Align
* 將人臉特徵點進行對齊，需要先定義對齊的座標，在 onnx arcface_inference.ipynb 裡的 Preprocess images 中可以看到。
* skimage 套件 transform.SimilarityTransform() 得到要變換的矩陣，然後進行對齊。
## Feature extraction
* 提取剛剛對齊後的人臉特徵，這邊示範使用 onnx ArcFace model。
* [InsightFace-REST 模型 arcface_r100_v1 ](https://github.com/SthPhoenix/InsightFace-REST)
* [Onnx](https://github.com/onnx/models/tree/master/vision/body_analysis/arcface)
    * Open Neural Network Exchange (ONNX) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves.
    * ONNX Runtime is a runtime accelerator for Machine Learning models
    
    
## 模型進行提取-人臉轉置
* 轉換 dtype 為 float32，最後進行 inference

## Create Database
* 這部分要將辨識的人臉資料寫進資料庫裡，這邊資料庫是使用 sqlite。
* 把Face Detection、Face Align、Feature extraction 寫成函數。然後將資料夾的圖片分別進行偵測、對齊、提取特徵後，再寫入資料庫裡。

## Face Recognition
* 資料庫裡的人臉特徵跟輸入照片進行比對，這邊使用 L2-Norm 來計算之間的距離。最後再設定 threshold，若 L2-Norm 距離大於 threshold 表示輸入照片不為資料庫裡的任何一個人；反之，L2-Norm 距離最小的人臉與輸入照片為同一個人。
## Demo影片
https://user-images.githubusercontent.com/12774427/132973166-619e5696-2bfb-4f06-9ae4-fa0117a13f18.mp4


