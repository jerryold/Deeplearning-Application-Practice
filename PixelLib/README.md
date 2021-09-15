--
# Instance Segmentation using Mask-RCNN

## Package
* 需使用套件conda輸出的freeze.yml即可使用
## 套件介紹
### PixelLib
* 最新的PixelLib可以在圖像、視頻中提取分割目標；以及利用coco檢測模型來實現使用者的分割類別。提供兩種圖像分割：語意分割和實例分割

### Mask_rcnn
* MaskRCNN 是何愷明基於以往的 faster rcnn 架構提出的新的卷積網絡，一舉完成了 object instance segmentation。該方法在有效地目標的同時完成了高質量的語義分割。文章的主要思路就是把原有的 Faster-RCNN 進行擴展，添加一個分支使用現有的檢測對目標進行並行預測。
    
```
程式碼補充
show_bboxes: This is the parameter that shows segmented objects with bounding boxes. If it is set to false, it shows only the segmentation masks.

frames_per_second: This is the parameter that sets the number of frames per second for the saved video file. In this case it is set to 5, i.e the saved video file would have 5 frames per second.

extract_segmented_objects: This is the parameter that tells the function to extract the objects segmented in the image and it is set to true.

save_extracted_objects: This is an optional parameter for saving the extracted segmented objects.

output_video_name: This is the name of the saved segmented video.
```

## 檔案介紹
* [mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases/tag/v2.0)-所使用的Mask_RCNN_COCO的h5檔案
* Instance Segmentation(Camera).ipynb:使用PixelLib flexible Python library 執行 all types of segmentation,並透過預訓練mask_rcnn_coco.h5,並透過opencv camera按下"q"鍵拍照,並回傳處理後照片

* Instance Segmentation(Movie).ipynb:使用PixelLib flexible Python library 執行 all types of segmentation,並透過預訓練mask_rcnn_coco.h5,並透過opencv camera按下"q"鍵將剛剛錄影影片存取後,並回傳處理後影片

* Instance Segmentation(Real time).ipynb:使用PixelLib flexible Python library 執行 all types of segmentation.並透過預訓練mask_rcnn_coco.h5 來偵測物體,並可即時透過opencv camera real time來偵測物體

## 範例圖如下
### Instance Segmentation(Camera)
![](https://i.imgur.com/JqzAP4A.jpg)
![](https://i.imgur.com/bPs8xyg.jpg)
![](https://i.imgur.com/NeBlPUr.jpg)

### Instance Segmentation(Video)
https://user-images.githubusercontent.com/12774427/133398232-2214fe84-acc9-45fe-be2b-c9699e0ed6f2.mp4










