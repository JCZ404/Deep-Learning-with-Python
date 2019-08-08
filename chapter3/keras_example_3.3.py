"""
(1)学会在当数据样本特别小的时候如何实现对模型的评价：
之前我们的策略是在训练数据中留出一个验证集，那是因为样本量足够大，而当样本量比较小的时候
验证集将会很小，导致验证分数会有很大的波动：即验证集的划分方式很可能导致验证分数具有很大的方差。
这样就无法实现对模型的准确的评价。
--->解决办法：k折交叉验证，将数据划分为k个分区，实例化k个模型，在k-1个分区上训练，在剩下的1个分区上进行验证，最后模型的
验证分数等于所有验证分数的平均值。
(2)评价模型是为了选择一个比较好的参数，为建造最后的模型做准备

"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from keras import layers
from keras import models

#加载数据：最好提前搞清楚数据的组织形式，结构，特点
(train_data,train_labels),(test_data,test_labels) = boston_housing.load_data()

#通过分析，可以发现这里的数据是不同类型的，成本型，效益型，不同的数据差距也很大，因此需要进行标准化：化为均值为0，方差为1的数据
mean = train_data.mean(axis = 0)        #在0轴(样本轴)上求均值
std = train_data.std(axis = 0)          #标准差
#数据标准化
train_data = train_data-mean            #这里有数据的广播运算
train_data = train_data/std

#对测试数据也要进行标准化，不过需要注意的是：在测试数据上进行的标准化用的均值和标准差只能是训练数据的，不能用测试数据本身
test_data = test_data-mean
test_data = test_data/std

def build_model():
    """
    用来构建模型，实现多次模型的实例化
    """
    model = models.Sequential()
    model.add(layers.Dense(64,input_shape=(train_data.shape[1],),activation='relu'))
    model.add(layers.Dense(64,activation='relu'))

    #因为是回归模型，所以最后一层只有一个单元而且不用加激活函数
    model.add(layers.Dense(1))
    #编译时的参数就是最后进行模型的evalute返回的目标
    model.compile(optimizer='rmsprop',loss = 'mse',metrics=['mae'])
    return model

k = 4
num_val_samples = len(train_data)//k;   #确定分区的样本数
num_epochs = 500
all_score = []
all_mae_histories = []

for i in range(k):
    '''实例化k次模型'''
    print('processing fold #',i)
    #验证数据
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]   #验证数据，一个分区
    val_labels = train_labels[i*num_val_samples:(i+1)*num_val_samples]

    #训练数据：注意要将那些不连续的合并
    partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis = 0)
    partial_train_labels = np.concatenate([train_labels[:i*num_val_samples],train_labels[(i+1)*num_val_samples:]],axis= 0)

    #建立模型
    model = build_model()

    #训练模型
    #1，获得k次试验模型的评价得分
    #history = model.fit(partial_train_data,partial_train_labels,epochs=num_epochs,batch_size=1)

    #简单的只是每一折都在验证集上评价一次模型，得到k个分数
    # val_mse,val_mae = model.evaluate(val_data,val_labels)
    # all_score.append(val_mse)

    #2,记录k折平均下来，每一轮的状况
    #verbose:日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    history = model.fit(partial_train_data,partial_train_labels,epochs= num_epochs,batch_size=1,validation_data = (val_data,val_labels),verbose=0)

    #记录训练历史:history记录了训练时每一轮的表现
    mae_history = history.history['val_mean_absolute_error']         #平均绝对误差,是一个列表，包含一折所有轮的训练信息
    print(history.history.keys())
    all_mae_histories.append(mae_history)

#如何求多个轮次的绝对误差的平均值
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

def smooth_curve(points,factor = 0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            #将每个数据替换为前面数据点的指数移动平均
            previous = smoothed_points[-1]   #列表添加的顺序，最后一个为最新添加的
            smoothed_points.append(previous*factor+point*(1-factor))  #指数平均，光滑处理
        else:
            #第一个点不处理
            smoothed_points.append(point)
    return smoothed_points

#对数据进行光滑处理,去掉前10个点
smooth_mae_history = smooth_curve(average_mae_history[10:])


plt.figure(1)
plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')

plt.figure(2)
plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

"""
上面所做的工作只是为了在训练时很好的评价这个模型，看看这个模型随着一些参数的改变，它的性能是如何变化的，目的是
为了提供建造模型时的一些重要参数的选择，如epochs，batch_size,知道这些参数取什么值的时候模型性能最好，就能建造出
最后的适用的模型
"""
#最终的生产模型
model = build_model()
model.fit(train_data,train_labels,batch_size=8,epochs=80,verbose=0)  #这些参数就是通过上面的模型的评价得到的
(test_mse_score,test_mae_score) = model.evaluate(test_data,test_labels)
print(test_mse_score,test_mae_score)