"""
Recurrent net to deal with time sequence:the temperature data recorded by the sensor
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import  Sequential
from keras.layers import Dense,LSTM,Dropout,Flatten,GRU
from keras.optimizers import RMSprop

#Visit Data
data_dir = "/home/zhangjc/PycharmProjects/Deep-Learning-with-Python/jena_climate/"
fname = os.path.join(data_dir,"jena_climate_2009_2016.csv")  #the timesteps is 10 minutes and there 14 features


f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')  #header and lines is list
lines = lines[1:]         #only keep the part of data
print(header)
print(len(header))
print(len(lines))

#Analyze Data
#transform the data into the numpy array
float_data = np.zeros((len(lines),len(header)-1))
for i,line in enumerate(lines):
    data = [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = data
print(float_data.shape)

# #View Temperature
# #1 day is 144 timesteps,1 hour is 6 timesteps
# temp = float_data[:,1]
# plt.figure(1)                       #view all the temperature data
# plt.plot(range(len(temp)),temp)
# plt.figure(2)
# plt.plot(range(1440),temp[:,1440])  #view the past 10 days temperature data
# plt.show()

#Step1:Prepare Data
#(1)normalize the data,make all of the data into the same interval
#(2)make a generator to input the former float array and product the data batch
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

#generate data
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,lookback=lookback,delay= delay,min_index=0,max_index=200000,shuffle = True,step=step,batch_size=batch_size)
val_gen = generator(float_data,lookback = lookback,delay= delay,min_index= 200001,max_index=300000,step = step,batch_size = batch_size)
test_gen = generator(float_data,lookback= lookback,delay = delay,min_index= 300001,max_index=None,step = step,batch_size= batch_size)
#test_gen = test_generator(float_data,lookback= lookback,delay = delay,min_index= 300001,max_index=None,step = step,batch_size= batch_size)

val_steps = (300000-200001-lookback)//batch_size         #the number of extracting data for get the entire data
test_steps = (len(float_data)-300001-lookback)//batch_size

#Step2:Build a standard result based on common sense
#this result can be the orignal result,usually is very simple,if the result of deep-learning model is not better the original result,
#it can't prove the effectiveness of deep-learnling model
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples,targets = next(val_gen)
        preds = samples[:,-1,1]              #here we assume the temperature after 24 hours is always equal to the temperature of now
        mae = np.mean(np.abs(preds-targets)) #mae
        batch_maes.append(mae)
    print(np.mean(batch_maes))
    return np.mean(batch_maes)

base_mae = evaluate_naive_method()
celsius_mae = base_mae*std[1]   #std[1] on behalf of std of temperature,celsius_mae is the difference of temperature

#Step3:Build a Deep-Learning model to improve the celsius_mae
#test1:use Dense Net
# model = Sequential()
# model.add(Flatten(input_shape=(lookback//step,float_data.shape[-1])))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(1))         #regression problem,don't need a activation function
#
# #test2:use GRU Net
# model = Sequential()
# model.add(GRU(32,input_shape=(None,float_data.shape[-1])))   #pay attention to the shape of input tensor
# model.add(Dense(1))

#test3:use a Dropout GRU Net
#the net with dropout should take longer time to reach convergence so when model fit,we need more epochs
model = Sequential()
model.add(GRU(32,dropout=0.2,recurrent_dropout=0.2,input_shape=(None,float_data.shape[-1])))
model.add(Dense(1))

#test4:use Recurrent Layer Stacking
# model = Sequential()
# model.add(GRU(32,dropout=0.2,recurrent_dropout=0.2,return_sequences=True,input_shape=(None,float_data.shape[-1]))) #should return the sequence
# model.add(GRU(64,activation='relu',dropout=0.1,recurrent_dropout=0.5))   #for get the sequence,here need a activation
# model.add(Dense(1))

#Step4:Compile and Train model
model.compile(optimizer=RMSprop(),loss = 'mae')
history = model.fit_generator(train_gen,epochs=40,steps_per_epoch=500,validation_data=val_gen,validation_steps=val_steps)
model.save("Temperature_Prediction")

#Step5:Visualize the result
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
