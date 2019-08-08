"""
Recurrent net to deal with text sequence:SimpleRNN,LSTM,Two-Way LSTM
(1)the three major factor:input,state,output
(2)recurrent nets are usually used to deal the sequence,so we must know exactly the structure of input,and
how to organize the input data
"""


import numpy as np
import matplotlib.pyplot as plt
from keras.layers import SimpleRNN,Dense,Embedding,LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import  Sequential
from keras.layers import Bidirectional   #this layer is uesed to build a two-way RNN

##simple RNN with numpy
# timesteps = 100
# input_features = 32
# output_features = 64
#
# inputs = np.random.random((timesteps,input_features))
#
# state_t = np.zeros((output_features,))
#
# W = np.random.random((output_features,input_features))
# U = np.random.random((output_features,output_features))
# b = np.random.random((output_features,))
#
# successive_outputs = []
# for input_t in inputs:
#     output_t = np.tanh(np.dot(W,input_t)+np.dot(U,state_t)+b)
#     successive_outputs.append(output_t)
#
#     state_t = output_t
#     # successive_outputs is a array,use stack to transform it to a tensor with two axis
# final_output_sequence = np.stack(successive_outputs,axis=0)  #(timesteps,output_features)


#Simple-RNN in keras
#simple RNN in keras is a layer,it can receive a batch sequence,like(batch_size,timesteps,input_features)
#it can return two results:
#(1)return the states of each timesteps:(batch_size,timesteps,output_features)
#(2)return the output of final timesteps:(batch_size,output_features)

max_features = 1000   #the number of word as features
maxlen = 500          #change the len of each comments into 500 words
batch_size = 32

#Step1:Prepare Data
#the most important thing is you must know the data structrue and what's it mean
print("Loading Data...")

#positive sequence
(input_train,y_train),(input_test,y_test) = imdb.load_data(num_words= max_features)#input_data:(num_sampels,length_each_comments) 2-dims-list
#negative sequence
#here we just reverse each comments,not the samples,so the labels shouldn't to be reversed
input_trian = [x[::-1]for x in input_train]
input_test = [x[::-1]for x in input_test]

print(len(input_train),'train sequence')
print(len(input_test),'test sequence')

print("Padding Sequence (samples x time")
input_train = sequence.pad_sequences(input_train,maxlen=maxlen)   #use sequence to cut or pad
input_test = sequence.pad_sequences(input_test,maxlen=maxlen)
print("input train shape:",input_train.shape)
print("input test shape:",input_test.shape)

##Step2:Build Model
#deal the text,we should use the Embedding as the first layer
#SimpleRNN
# model = Sequential()
# model.add(Embedding(max_features,32))
# model.add(SimpleRNN(32))
# model.add(Dense(1,activation='sigmoid'))
#
# model.compile(optimizer='rmsprop',loss = 'binary_crossentropy',metrics=['acc'])
# #still make the train data into two parts
# history = model.fit(input_train,y_train,epochs=20,batch_size=128,validation_split=0.2)

#LSTM
#in theory,the result should reach 89%
# model = Sequential()
# model.add(Embedding(max_features,32))
# model.add(LSTM(32))
# model.add(Dense(1,activation='sigmoid'))

#Two-way LSTM
#in theory,this model will have a better result than the model with a simgle LSTM
model = Sequential()
model.add(Embedding(max_features,32))
model.add(Bidirectional(LSTM(32,dropout = 0.2,recurrent_dropout = 0.2)))#the Bidirectional will make a another LSTM to train on the revserse data
model.add(Dense(1,activation = 'sigmoid' ))


epochs = 20
model.compile(optimizer='rmsprop',loss = 'binary_crossentropy',metrics=['acc'])
#still make the train data into two parts
history = model.fit(input_train,y_train,epochs=epochs,batch_size=128,validation_split=0.2)
model.save("IDMB_classification.h5")

#Step3:Visualize Results
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

#Step4:Test Model
print("The accuracy of model:",model.evaluate(input_test,y_test))
