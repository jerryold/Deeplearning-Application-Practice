# The project is different practice for the deeplearning Application

1.Activation Comparision-比較傳統&新生代的激活函數效果
* 傳統:relu,leaky_relu,tanh,sigmoid
* 新生代:selu,swish,mish,bert版的gelu

2.Background Removal-基於Deeplab v3架構進行圖片去背效果訓練

3.Facerecognization(RetinaFace)-基於Retaina Face Model進行人臉辨識

4.Faceemotion(Deepface)-基於Deepface進行人臉情緒辨識,並透Opencv_haarcascade偵測人臉特徵(內含圖片辨識和攝影機即時辨識)

5.Faceemotion_MobileNetV2-基於Kaggle 提供的FER2013人臉訓練集,使用MobileNetV2框架進行進行人臉情緒辨識，並透Opencv_haarcascade偵測人臉特徵(圖片辯識)

6.PixelLib-基於PixelLiB套件分別撰寫Real time Camera,Video,Photo三種不同方式的Instance Segementation,並透過預處理好的mas_rcnn_coco(需自行去下載)訓練去偵測物體特徵