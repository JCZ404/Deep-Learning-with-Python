import numpy as np
import matplotlib.pyplot as plt
#解决print不输出的问题
from imp import reload
import sys
reload(sys)


from keras.datasets import imdb
from keras import regularizers
from keras import layers
from keras import models


(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words= 10000) #只保留训练数据中前10000个最常见的单词


#将整数序列转换为二进制矩阵
def vectorize_sequences(sequences,dimensions = 10000):
    results = np.zeros((len(sequences),dimensions))         #记得要加上中括号，或者还加一层括号才能正确创建数组
    for i,sequences in enumerate(sequences):
        results[i,sequences]=1
    return results

#将训练的特征数据向量化
x_train = vectorize_sequences(train_data)
x_test  = vectorize_sequences(test_data)

#这里的标签本身已经是整数序列了,所以对标签向量化较为简单
y_train = np.asarray(train_labels).astype('float32')
y_test  = np.asarray(test_labels).astype('float32')

#构建网络,用不同容量的网络进行测试，看看什么类型的网络不容易过拟合
model = models.Sequential()
model.add(layers.Dense(16,input_shape=(10000,),activation='relu'))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))              #二分类，最后一层应该用sigmoid函数输出一个标量

#尝试更小的模型
# model1 = models.Sequential()
# model1.add(layers.Dense(4,activation='relu',input_shape=(10000,)))
# model1.add(layers.Dense(4,activation='relu'))
# model1.add(layers.Dense(1,activation='sigmoid'))

#尝试使用正则化
# model1 = models.Sequential()
# model1.add(layers.Dense(16,activation='relu',input_shape=(10000,),kernel_regularizer=regularizers.l2(0.001)))
# model1.add(layers.Dense(16,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
# model1.add(layers.Dense(1,activation='sigmoid'))

#使用Dropout来降低过拟合
model1 = models.Sequential()
model1.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model1.add(layers.Dense(16,activation='relu'))
model1.add(layers.Dropout(0.5))
model1.add(layers.Dense(1,activation='sigmoid'))

#模型编译：二分类问题，损失函数设为二元交叉熵(也可以尝试mse均方误差)，监测的指标为精度
model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])
model1.compile(optimizer='rmsprop',loss = 'mse',metrics=['accuracy'])


#训练前的数据处理：将原始训练数据留出10000份作为验证集，用来测控模型在没有见过的数据上的精度
x_val = x_train[:10000]               #验证集
partial_x_train = x_train[10000:]     #真正用于训练的数据
y_val = y_train[:10000]              #验证集
partial_y_train = y_train[10000:]     #真正用于训练的数据

#训练:每训练512个样本更新一次权重,对所有样本迭代20轮，每一轮结束要计算一下精度和损失
history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
history1 = model1.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))


#分析训练历史数据：训练时训练数据的损失函数值，精度，以及在没有见过的数据---验证集上的损失函数，精度
history_dict = history.history
history_dict1 = history1.history
print(history_dict.keys())

#模型最终结果
results = model.evaluate(x_test,y_test)
results1 = model1.evaluate(x_test,y_test)
print("模型容量适中的结果：",results)
print('模型容量过小的结果：',results1)

#对历史数据可视化
#按训练的轮次来记录的历史数据,每迭代完一轮计算精度和损失
#训练数据的损失函数值
loss_value = history_dict['loss']       #验证集上的损失函数值
val_loss_value = history_dict['val_loss']  #损失函数值
loss_value1 = history_dict1['loss']
val_loss_value1 = history_dict1['val_loss']

epochs = range(1,len(loss_value)+1)

plt.figure(1)
plt.plot(epochs,loss_value,'bo',label = "Training loss")
plt.plot(epochs,val_loss_value,'b+',label = "Validation loss")
plt.title('Traing and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.plot(epochs,loss_value1,'bo',label = "Training loss")
plt.plot(epochs,val_loss_value1,'r+',label = 'Validation loss of smaller model')
plt.title('Training and validation loss with dropout model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#精度
# plt.figure(3)
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
#
# plt.plot(epochs,acc,'bo',label = 'Training accuracy')
# plt.plot(epochs,val_acc,'b',label = 'Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

plt.show()

