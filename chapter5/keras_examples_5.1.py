"""
卷积神经网络
"""
"""
A Simple CNN：the purpose of this program is to learn the basic theory of the convolution net,
and know the function of each of layers 
"""

#实例化一个简单的卷积神经网络实现手写数字识别
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation= 'relu',input_shape=(28,28,1)))    #input tensor:(image_height,image_width,image_channels)
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

#output tensor of above:(3,3,64)
model.add(layers.Flatten())                         #make the 3D tensor flatten to 1D
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

#train model on the convnet
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',loss= 'categorical_crossentropy',metrics= ['accuracy'])
model.fit(train_images,train_labels,epochs=5,batch_size = 64)