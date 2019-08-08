
"""
Keras实现新闻分类，多分类的问题
使用的是路透社的新闻数据，包含了46个不同的主题
主要关注的问题是对于多分类问题，对于数据标签是如何处理的
(1)转换为0-1独热编码
(2)直接只转换为整数张量，不过，相应的模型的损失函数应该做出相应的改变

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()


from keras.datasets import reuters
from keras.utils import to_categorical
from keras import layers
from keras import models

(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)

#尝试将索引数据解码为单词文本
word_index  = reuters.get_word_index()
reverse_word_index = dict([value,key] for (key,value) in word_index.items())
decoded_newswire = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])     #这里只解码训练集的第一条，用空格作为连接符

print(decoded_newswire)
print(len(train_data))
print(len(test_data))
print(test_labels[0])

#将数据向量化
#训练数据向量化
def vectorize_sequences(sequences,dimensions = 10000):
    results = np.zeros((len(sequences),dimensions))         #记得要加上中括号，或者还加一层括号才能正确创建数组
    for i,sequences in enumerate(sequences):
        results[i,sequences]=1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#标签数据向量化
#由于是多类别，所以需要利用0-1独热编码:每个标签为一个全零向量，只有标签索引对应元素为1
#如果不想转换为0-1编码，可以直接用整数张量，不过后面编译时，损失函数应该定义为sparse_categorical_crossentropy
def to_one_hot(labels,dimension = 46):     #注意维度只有46
    results = np.zeros((len(labels),dimension))    #先创建预先内存
    for i,label in enumerate(labels):
        results[i,label] = 1
    return results

y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

"""其实可以用keras里现成的方法"""
one_hot_train_labels = to_categorical(train_labels)
ont_hot_test_labels = to_categorical(test_labels)

#搭建网络
model = models.Sequential()
model.add(layers.Dense(128,input_shape=(10000,),activation='tanh'))
model.add(layers.Dense(128,activation='tanh'))
model.add(layers.Dense(46,activation='softmax'))         #最后一层的单元数要与输出类别数相同，多分类问题，最后一层激活函数用softmax

#模型编译
#多分类问题使用分类交叉熵作为损失函数
model.compile(optimizer='rmsprop',loss = 'categorical_crossentropy',metrics = ['accuracy'])

#留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

#训练模型
history = model.fit(partial_x_train,partial_y_train,epochs= 10,batch_size=512 ,validation_data=(x_val,y_val))


#将训练过程可视化
history_dict = history.history

#损失函数
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1,len(loss)+1)

#精度
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.figure(1)
plt.plot(epochs,loss,'bo',label = 'Traing loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title('Traing and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.plot(epochs,acc,'bo',label = "Train accuracy")
plt.plot(epochs,val_acc,'b',label = 'Validation accuracy')
plt.title('Traing and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

results = model.evaluate(x_test,y_test)
print(results)