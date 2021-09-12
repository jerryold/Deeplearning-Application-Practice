---
tags: Artifical Intelligence Practice&Readme
---

# 活化函數練習
## Trident介紹
* 引用trident的方法，語法如下，需要透過環境變數來指定使用框架，目前支持pytorch 1.2以上以及pytorch:1.4以上版本

* 本次為作者尹相志作者課程封裝的 API trident 中，只需要利用get_activation('活化函數名稱')即可回傳對應的活化函數


## 各個活化函數值域以及一階導數分布圖
* 只需要利用get_activation('活化函數名稱')即可回傳對應的活化函數，我們可以利用它結合Matplotlib 來繪製值域以及一階導數分布圖。


![](https://i.imgur.com/hotp5fD.png)
![](https://i.imgur.com/gD5xFsk.png)
![](https://i.imgur.com/sZGyvJg.png)


## 評測傳統活化函數與新的活化函數之間的效能差異
1. 計算時長(表示計算的開銷)
2. 跑 mnist 1000 minibatch 的訓練階段最佳accuracy以及最後10個批次的accuracy
3. 使用一樣的數據進行推論
4. 使用加入大量噪聲的數據進行推論(確認跑出來模型的通用性)
5. 檢視訓練過程的梯度與權重分布
6. 梯度為零比率

* 加入高度噪聲的數據，只需要在noise_dataset.image_transform_funcs中多加入個add_noise(intensity=0.3)，表示會加入30% 強度的隨機噪音

* 本次比較傳統和新生代活化函數之間的差異

| 傳統         |    新生代    |
| ------------ |:------------:|
| relu,        |    selu    |
| leaky_relu， |    swish     |
| tanh         |     mish     |
| sigmoid      | bert版的gelu |

* test_activity_function(測試function)

 ### 計算跑 10 萬次的總時間
```
  data =to_tensor(np.random.standard_normal((100000,1))) 
    start = time.time()
    results=[act_func(data[i]) for  i  in range(100000)]
    sec = time.time() - start
    print('{0:.6f} sec'.format(sec))
    performance_dict[act]['跑 10 萬次的總時間']=sec
```

 ### 跑1000 次，過程中的權重變化與梯度變化都保留
`plan.only_steps(num_steps=1000,collect_data_inteval=10,keep_weights_history=True,keep_gradient_history=True )`

### 乾淨數據與噪聲數據進行推論
```

    net.eval()
    accuracys=[]
    for i in range(100):
        input,target=data_provider.next()
        input,target=to_tensor(input),to_tensor(target)
        accuracys.append(to_numpy(accuracy(net(input),target)))
    accuracys=np.asarray(accuracys)
        
    print('正常樣本 accuracy {0:.4%}  DIFF: {1:.4%} '.format(accuracys.mean(),accuracys.mean()-performance_dict[act]['最後10次 metrics']))
    performance_dict[act]['正常樣本 accuracy DIFF']=accuracys.mean()-performance_dict[act]['最後10次 metrics']
    
    
    noise_accuracys=[]
    for i in range(100):
        input,target=noise_data_provider.next()
        input,target=to_tensor(input),to_tensor(target)
        noise_accuracys.append(to_numpy(accuracy(net(input),target)))
    noise_accuracys=np.asarray(noise_accuracys)
        
    print('噪聲樣本 accuracy {0:.4%}  DIFF: {1:.4%} '.format(noise_accuracys.mean(),noise_accuracys.mean()-performance_dict[act]['最後10次 metrics']))
    performance_dict[act]['噪聲樣本 accuracy DIFF']=noise_accuracys.mean()-performance_dict[act]['最後10次 metrics']
```
### 繪製梯度與權重分布

```
weights_history=plan.training_items[0].weights_history
gradients_history=plan.training_items[0].gradients_history
    
grads=[]
weights=[]
    for i in range(len(gradients_history)):
        grads.append(gradients_history[i].value_list[0].reshape([-1]))

    for i in range(len(weights_history)):
         weights.append(weights_history[i].value_list[0].reshape([-1]))
    
    grads=np.asarray(grads)
    weights=np.asarray(weights)
```
    

####  <font color="red">relu</font>
![](https://i.imgur.com/TtGIqev.png)
![](https://i.imgur.com/UTXebBG.png)

####  <font color="red">leaky_relu</font>
![](https://i.imgur.com/1Yaclim.png)
![](https://i.imgur.com/hJD7CUO.png)

####  <font color="red">sigmoid</font>
![](https://i.imgur.com/qlEKkRm.png)
![](https://i.imgur.com/BDyx6SN.png)


####  <font color="red">tanh</font>
![](https://i.imgur.com/JmXO9Gm.png)
![](https://i.imgur.com/VqntIS0.png)


####  <font color="red">selu</font>
![](https://i.imgur.com/cbJqUhD.png)
![](https://i.imgur.com/ebernE5.png)



####  <font color="red">swish</font>
![](https://i.imgur.com/B5JGt4e.png)
![](https://i.imgur.com/q5u5sll.png)


####  <font color="red">mish</font>
![](https://i.imgur.com/09RgNhr.png)
![](https://i.imgur.com/heSdSKU.png)


####  <font color="red">gelu</font>
![](https://i.imgur.com/QoZUUxB.png)
![](https://i.imgur.com/hX3j129.png)

####  <font color="red">SIREN</font>
![](https://i.imgur.com/TX3xvt3.png)
![](https://i.imgur.com/F5SReg5.png)


## 總結
![](https://i.imgur.com/fUHaS53.png)


* gelu的計算開銷是遠大於其他活化函數的
* 新生代的活化函數整體上看來，泛化能力是排後段班的
* z軸高度是根據圖表的最大值設定的，因此如果你看到有些圖看起來梯度很稀疏，幾乎看不到紅色區域，那就是它的梯度都集中在特定點(通常是0)，你可以看到relu是全部中梯度為零比例最高者，leaky_relu看起來很有效的緩解了relu梯度為零的缺陷，新生代的活化函數的梯度為零比率看起來表現尚可。

## 建議
* 別再使用relu了，如果你習慣傳統活化函數的低計算開銷，建議改成leaky_relu
* 如果你想嘗試新的活化函數，selu看起來是比較好的選擇

