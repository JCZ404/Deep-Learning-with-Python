"""
用卷积神经网络处理序列：一维卷积神经网络
适用：文本分类，时间序列预测
特点：计算代价比RNN小
"""
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras import layers

'''============================================Part-1:IMDB Classification================================================'''
"""
#Step1:Prepare Data
max_features = 1000
max_len = 500
print("Loading data...")
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words= max_features)

print("Padding data...")
x_train = sequence.pad_sequences(x_train,maxlen = max_len)
x_test = sequence.pad_sequences(x_test,maxlen = max_len)

print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)


#Step2:Build Model
#Conv1D receive the tensor with the shape of(samples,timesteps,features),here it's (samples,input_length,output_dim)
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))  #the length of convolution window is 7,use this 7-len window slide on the input_length
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
#Step3:Compile and Train Model
model.compile(optimizer=RMSprop(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])

history = model.fit(x_train, y_train,epochs=6,batch_size=128,validation_split=0.2)

model.save("Conv1D_imdb.h5")

#Step4:Visualize Results
epochs = 6
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']

x = range(1,epochs+1)
plt.figure(1)
plt.plot(x, loss,'bo', label='Train loss')
plt.plot(x, val_loss, 'b', label='Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Train and Validation Loss')
plt.legend()

plt.figure(2)
plt.plot(x,acc, 'bo', label='Train accuracy')
plt.plot(x, val_acc, 'b', label='Validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Train and Validation accuracy')
plt.legend()
plt.show()
"""
"""===========================================Part-2:Jena Climate Prediction==========================================="""
#Step1:Prepare data
#Visit Data
data_dir = "/home/zhangjc/PycharmProjects/Deep-Learning-with-Python/jena_climate/"
fname = os.path.join(data_dir,"jena_climate_2009_2016.csv")  #the timesteps is 10 minutes and there 14 features

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')  #header and lines is list
lines = lines[1:]             #only keep the part of data

#Analyze Data
float_data = np.zeros((len(lines),len(header)-1))
for i,line in enumerate(lines):
    data = [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = data

#data normalization
mean = float_data[:200000].mean(axis = 0)
float_data -= mean
std = float_data[:200000].std(axis = 0)
float_data /= std

#data generator
def generator(data,lookback,delay,min_index,max_index,shuffle = False,batch_size = 128,step = 6):
    #the range of data should be limited
    if max_index is None:
        max_index = len(data) - delay -1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index+lookback,max_index,size=batch_size)
        else:
            if i+batch_size>=max_index:
                i = min_index+lookback
            rows = np.arange(i,min(i+batch_size,max_index))
            i += len(rows)
        samples = np.zeros((len(rows),lookback//step,data.shape[-1]))
        targets = np.zeros((len(rows),))

        for j,row in enumerate(rows):
            indices = range(rows[j]-lookback,rows[j],step)   #for each of 128 rows,we take lookback previous data as the feature,it is a tensor
            samples[j] = data[indices]                     #data is 2-dims tensor,samples[j] is also 2-dims tensor
            targets[j] = data[rows[j]+delay][1]            #the target is the temperature after delay timesteps
        yield samples,targets                              #samples:(batch_size,lookbach//step,num_of_features) targets:(batch_size,)

#Test1:use Conv1D only
#this model perform bad,but it's faster than the LSTM
#generate data
# lookback = 1440 #lookback 10 days
# step = 6
# delay = 144
# batch_size = 128

#Test2:combination Conv1D with RNN
#Conv1D is faster and can deal with the longer sequence,while RNN is sensetive with the order
lookback = 720   #look back 5 days
step = 3        #extract more samples
delay = 144
batch_size = 128


train_gen = generator(float_data,lookback=lookback,delay= delay,min_index=0,max_index=200000,shuffle = True,step=step,batch_size=batch_size)
val_gen = generator(float_data,lookback = lookback,delay= delay,min_index= 200001,max_index=300000,step = step,batch_size = batch_size)
test_gen = generator(float_data,lookback= lookback,delay = delay,min_index= 300001,max_index=None,step = step,batch_size= batch_size)
#test_gen = test_generator(float_data,lookback= lookback,delay = delay,min_index= 300001,max_index=None,step = step,batch_size= batch_size)

val_steps = (300000-200001-lookback)//batch_size         #the number of extracting data for get the entire data
test_steps = (len(float_data)-300001-lookback)//batch_size

#Step2:Build Model
#Test1:use Conv1D only
# model = Sequential()
# model.add(layers.Conv1D(32,5,activation = 'relu',input_shape = (None,float_data.shape[-1])))
# model.add(layers.MaxPooling1D(3))
# model.add(layers.Conv1D(32,5,activation = 'relu'))
# model.add(layers.MaxPooling1D(3))
# model.add(layers.Conv1D(32,5,activation = 'relu'))
# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(1))

#Test2:combination Conv1D with RNN
#this result may not be better than the simgle GRU or LSTM,but it's faster
model = Sequential()
model.add(layers.Conv1D(32,5,activation = 'relu',input_shape = (None,float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32,5,activation = 'relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.LSTM(32,dropout = 0.2,recurrent_dropout = 0.2))
model.add(layers.Dense(1))

#Step3:Compile and Train Model
model.compile(optimizer=RMSprop(),loss='mae')

history = model.fit_generator(train_gen,epochs=20,steps_per_epoch=500,validation_data=val_gen,validation_steps=val_steps)
model.save("Temperature_Prediction_Conv1D.h5")

#Step4:Visualize the result
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

x = range(1,len(loss)+1)
plt.figure(1)
plt.plot(x, loss,'bo', label='Train loss')
plt.plot(x, val_loss, 'b', label='Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Train and Validation Loss')
plt.legend()

plt.show()
