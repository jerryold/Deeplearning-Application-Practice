#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
from IPython import display

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({'font.size': 10})#指定字體大小

# matplotlib.rcParams[‘figure.figsize’]#图片像素
# matplotlib.rcParams[‘savefig.dpi’]#分辨率
# plt.savefig(‘plot123_2.png’, dpi=200)#指定分辨率


# # 活化函數大清點 (pytorch)

# 支援python 版本: 3.5以上
# 支援pytorch: 1.4以上
這個實作是關於活化函數的理解。在神經元的底層結構中，活化函數扮演著將收到的信號賦予非線性特性的關鍵角色，也因此活化函數雖然不是太顯眼，但是仍有不少研究者推出新的活化函數希望可以藉由活化函數來提升模型效度。
# In[3]:


import gc
import glob
import os
import time
import cv2
os.environ['TRIDENT_BACKEND'] = 'pytorch'
# !pip install tridentx --upgrade
import trident as T
from trident import *
from trident.layers.pytorch_activations import __all__


在我這次為課程封裝的 API trident 中，只需要利用get_activation('活化函數名稱')即可回傳對應的活化函數，我們可以利用它結合Matplotlib 來繪製值域以及一階導數分布圖。
# In[4]:



fig = plt.figure(figsize=(16, 16))
plt.ion()
plt.subplots_adjust(wspace=0.4, hspace=0.6)
n = 1
items = __all__[:-1]  # 取得活化函數名稱清單

for k in items:
    if k not in ('PRelu','CRelu'):
        try:
            act_fn = get_activation(k)
            x = np.arange(-10, 10, 0.1).astype(np.float32) # 產生從-10到10之間，每隔0.1一筆的向量(x)
            tensor_x = expand_dims(to_tensor(x),0)  # 向量轉tensor
            y = to_numpy(act_fn(tensor_x))[0]  # 計算向量經過活化函數的結果(y)
            ax1 = fig.add_subplot(7, 4, n)  # 宣告7*4個子圖
            ax1.plot(x, y)  # 將x,y繪製圖表
            ax1.plot(x[1:], np.diff(y) / (np.diff(x) + 1e-8), ls=':')  # np.diff在計算隔一筆的差異。將y的微小變化除以x的微小變化就是梯度
            ax1.set_title(k)

        except Exception as e:
            print(e)
            pass
        n += 1
display.display(fig)  # 將圖表顯示在notebook上
plt.close(fig)  # 把figure關閉，免得出現一堆關不掉的窗口


# 接下來我們想要評測一下幾個傳統活化函數與新的活化函數之間的效能差異，我們將從幾點來評估它:
# 1.  計算時長(表示計算的開銷)
# 2. 跑 mnist 1000 minibatch 的訓練階段最佳accuracy以及最後10個批次的accuracy
# 3. 使用一樣的數據進行推論
# 4. 使用加入大量噪聲的數據進行推論(確認跑出來模型的通用性)
# 5. 檢視訓練過程的梯度與權重分布
# 6. 梯度為零比率

# 下面的語法是使用trident API 讀取mnist數據集
# 你也可以換成T.load_mnist('fashion-mnist')來改讀取fashion mnist

# In[5]:


data_provider=T.load_mnist('fashion-mnist')   #取得fashion-mnist數據集

im,label=data_provider.next() #調用一次以獲取一個minibatch的數據與標籤
print(im.shape)
im=array2image(np.concatenate(im[:8],axis=2)).resize((64*8,64))
#取8筆，把形狀reshapet成(8,28,28)且反正規化，並且沿著寬的方向(axis=1)疊合，並且把圖片尺寸放大到(64*8,64)

data_provider.image_transform_funcs=[Normalize(127.5,127.5)]  #利用減127.5除以127.5的方式來正規化圖片向量(值域會落在正負1之間)
im


# 若是要產生加入高度噪聲的數據，只需要在noise_dataset.image_transform_funcs中多加入個add_noise(intensity=0.3)，表示會加入30% 強度的隨機噪音，我們將它產生的數據輸出看一下，會是如下圖的結果，使用他的目的在於用乾淨的數據訓練，看看它第一次遇到那麼髒的數據的效度會有多大的程度衰退。

# In[6]:


noise_data_provider=T.load_mnist('fashion-mnist')#取得mnist數據集
noise_data_provider.image_transform_funcs=[AddNoise(intensity=0.3)]
noise_im,noise_label=noise_data_provider.next() #調用一次以獲取一個minibatch的數據與標籤
print(noise_im.shape)
noise_im=array2image(np.concatenate(noise_im[:8],axis=2)).resize((64*8,64))
#取8筆，把形狀reshapet成(8,28,28)且反正規化，並且沿著寬的方向(axis=1)疊合，並且把圖片尺寸放大到(64*8,64)
noise_data_provider.image_transform_funcs=[AddNoise(intensity=0.3),Normalize(127.5,127.5)]#除了正規化之外`,，額外添加30%噪音    
noise_im


# 我們這次選擇測試用的活化函數包括了較為傳統的relu, leaky_relu，tanh以及sigmoid，同時也比較了幾個新生代的活化函數，包括了selu,swish,mish以及bert版的gelu。 

# In[7]:


activations=['relu','leaky_relu','sigmoid','tanh','selu','swish','mish','gelu','SIREN']
performance_dict=OrderedDict()


# 接下來就把要測試的目標包裝成測試函數

# In[8]:



def test_activity_function(act):
    performance_dict[act]=OrderedDict()
    #清掉記憶體垃圾以避免影響效能
    gc.collect()
    act_func=get_activation(act)
    print(act)
    
    #計算跑 10 萬次的總時間
    data =to_tensor(np.random.standard_normal((100000,1))) 
    start = time.time()
    results=[act_func(data[i]) for  i  in range(100000)]
    sec = time.time() - start
    print('{0:.6f} sec'.format(sec))
    performance_dict[act]['跑 10 萬次的總時間']=sec

    #建 mnist模
    net=Sequential(
    Flatten(),
    Dense(128,use_bias=False,activation=act),
    Dense(64,use_bias=False,activation=act),
    Dense(10,use_bias=False,activation=None),
    SoftMax())
    
    model=Model(input_shape=(1,28,28),output=net)    .with_optimizer(optimizer='SGD',lr=1e-3)    .with_loss(CrossEntropyLoss)    .with_metric(accuracy)
    
    plan=TrainingPlan()    .add_training_item(model)    .with_data_loader(data_provider)    .with_batch_size(128)    .print_progress_scheduling(100,unit='batch')
    #讓他跑1000 次，過程中的權重變化與梯度變化都保留
    plan.only_steps(num_steps=1000,collect_data_inteval=10,keep_weights_history=True,keep_gradient_history=True )
    
    loss_indexs,loss_values=model.training_context['losses'].get_series('total_losses')
    accuracy_indexs,accuracy_values=model.training_context['metrics'].get_series('accuracy')
    print('最低 loss {0:.4e}  最佳 metrics {1:.4%}  最後10次 metrics {2:.4%}'.format(np.array(loss_values).min(),
                                                                              np.array(accuracy_values).max(),
                                                                              np.array(accuracy_values)[-10:].mean()))
    performance_dict[act]['最佳 metrics'] =np.array(accuracy_values).max()
    performance_dict[act]['最後10次 metrics'] =np.array(accuracy_values)[-10:].mean()
    
    
    # 分別使用乾淨數據與噪聲數據進行推論
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
    
    #繪製梯度與權重分布
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
   


# In[9]:


test_activity_function('relu')


# In[10]:


test_activity_function('leaky_relu')


# In[11]:


test_activity_function('sigmoid')


# In[12]:


test_activity_function('tanh')


# In[13]:


test_activity_function('selu')


# In[14]:


test_activity_function('swish')


# In[15]:


test_activity_function('mish')


# In[16]:


test_activity_function('gelu')


# In[17]:


test_activity_function('SIREN')


# 首先要來看的是這幾個活化函數的計算開銷，我們紀錄每個活化函數各執行1萬次的執行時間長度，我們可以發現gelu的計算開銷是遠大於其他活化函數的。接下來我們想要看看這些模型實際面對測試數據以及面臨到數據品質極差的樣本時，是不是具有足夠的泛化能力。這邊可以看到新生代的活化函數整體上看來，泛化能力是排後段班的。  
# 
# 接下來我們將收集的梯度與權重歷程(我們是設定collect_data_inteval=100，即每100批次收集一次數據)，首先是看梯度分布的狀況，由於這個直方圖的z軸高度是根據圖表的最大值設定的，因此如果你看到有些圖看起來梯度很稀疏，幾乎看不到紅色區域，那就是它的梯度都集中在特定點(通常是0)，你可以看到relu是全部中梯度為零比例最高者，leaky_relu看起來很有效的緩解了relu梯度為零的缺陷，新生代的活化函數的梯度為零比率看起來表現尚可。
# 
# 整體來看，新舊各有優點，老實說活化函數的影響其實低於優化器或者是權重初始化，若要我給個結論，我會建議的是：「別再使用relu了，如果你習慣傳統活化函數的低計算開銷，建議改成leaky_relu，如果你想嘗試新的活化函數，selu看起來是比較好的選擇」

# In[18]:


import pandas
#將performance_dict轉成表格
pd=pandas.DataFrame.from_dict(performance_dict)
pd

