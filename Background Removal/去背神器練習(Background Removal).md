---
tags: Artifical Intelligence Practice
---

## Trident介紹
* trident的原意是三叉戟，象徵的我們之前一貫維持一個實作三種框架的平行代碼的風格(cntk, pytorch, tensorflow)，三種框架各自有不同的設計概念以及實現過程都有許多繁瑣的坑，而trident的目的就是希望能夠讓大家少掉進坑裡，盡量讓繁瑣的細節隱藏，讓訓練與推論pipline盡量抽象化進而可以變成容易套用的流程，目前trident主要是在pytorch上開發，在trident中可以使用類似keras的模式來設計網路結構，同時讓訓練流程變得簡潔卻強大，而且非常容易修改與維護

* 引用trident的方法也很簡單，語法如下，需要透過環境變數來指定使用框架，目前支持pytorch 1.2以上以及tensorflow 2.0以上版本
```
import os os.environ['TRIDENT_BACKEND'] = 'pytorch' 
import trident as T from trident import * '
```


## 在本次練習實作中，數據集將都會以原作者提供的Trident API來實現。本次使用的數據集是吳恩達創辦的supervisely所發布的人體分割數據集Supervisely Person，在我開發的trident api中可以透過關鍵字"people"下載，裡面附的是將原圖稍作縮小的精簡版，已經足以用來訓練本次模型。若是各位對於原始數據感興趣，您也可以到 https://supervise.ly/ 即可下載取得圖資。

> 可以直接使用pip安裝trident pip install tridentx --upgrade

# 去背神器練習(Background Removal)
    -支援python 版本: 3.5以上
    -支援pytorch版本 : 1.4以上

* 使用Deeplab v3架構
    * Deeplab v3+中的主骨幹範圍總共將原圖下採樣4次，等於是長寬縮小1/16倍，因此要將EfficientNet裁切成適合Deeplab v3+使用的話，需要能夠找到圖像長寬縮小1/16倍的主骨幹範圍。在trident API中，會自動搜索圖像長寬縮小1/4倍(224x224縮小到56x56)以及1/16(224x224縮小到14x14)之處，來作為低層級特徵(low-level features)以及連接ASPP的高階特徵使用。對應到生成的Deeplab結構分別為backbond1.x以及backbond2.x。為了節省計算量，最後下採樣的解碼器，我使用depthwise卷積來替代標準的卷積。


* 模型設定
    * with_optimizer 設定優化器為Ranger
    * with_loss(DiceLoss) 加入Dice loss損失函數(類別層級)
    * with_loss(CrossEntropyLoss,2)加入交叉熵作為損失函數(像素層級)
    * with_loss(IouLoss,1)加入iou損失作為損失函數(類別層級) 直接針對iou優化


* 使用epoch 10 後train最終結果
![](https://i.imgur.com/kdKsssi.png)
![](https://i.imgur.com/IDEaCwD.png)
![](https://i.imgur.com/J9aQGKc.png)

