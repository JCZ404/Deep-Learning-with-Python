"""
使用LSTM生成文本
"""
import numpy as np
import matplotlib.pyplot as plt
import keras
import random
import sys
from keras import models
from keras import Sequential
from keras.layers import LSTM,Dense

#Get the softmax temperature to random sampling
#Not just choose the biggest probability results,we use the temperature to recalculate the distribution
def reweight_distribution(original_distribution,temperature = 0.5):  #temperature is the entropy,standing the degree of chaos
    #original_distribution is a numpy array
    distribution = np.log(original_distribution)/temperature
    distribution = np.exp(distribution)
    return distribution/np.sum(distribution)

#Step1:Prepare Data
#path = keras.utils.get_file('nietzsche.txt',origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
path = "/home/zhangjc/.keras/datasets/nietzsche.txt"
text = open(path).read().lower()
print('Corpus length:', len(text))

max_len = 60   #length of each sequence is 60
step = 3       #the interval is 3 chars
sentences = []
next_chars = []

for i in range(0,len(text)-max_len,step):
    sentences.append(text[i:i+max_len])
    next_chars.append(text[i+max_len])
print("Number of sequences is:",len(sentences))
chars = sorted(list(set(text)))    #the sorted unique chars list
print("Number of unique chars is:",len(chars))
char_indices = dict((char,chars.index(char))for char in chars)
print("Vectorization...")
#one-hot encode the sequence
x = np.zeros((len(sentences),max_len,len(chars)),dtype=np.bool)   #shape of x:(num_of_sentences,length_of_each_sentence,len(unique_chars))
y = np.zeros((len(sentences),len(chars)),dtype=np.bool)
for i,sentence in enumerate(sentences):
    for j,char in enumerate(sentence):
        x[i,j,char_indices.get(char)]=1
        y[i,char_indices.get(next_chars[i])]=1

#Step2:Build Model
model = Sequential()
model.add(LSTM(128,input_shape=(max_len,len(chars))))
model.add(Dense(len(chars),activation='softmax'))

#Step3:Complie and Train Model
optimizer = keras.optimizers.RMSprop(lr = 0.01)
model.compile(optimizer='rmsprop',loss = 'categorical_crossentropy',metrics=['acc'])

#Step4:Sample the index of next char
def Sample(preds,temperature = 1.0):
    preds = np.array(preds).astype('float64')
    preds = np.log(preds)/temperature
    preds_exp = np.exp(preds)
    preds = preds_exp/np.sum(preds_exp)
    probas = np.random.multinomial(1,preds,1)    #multinomial distribution sampling once?
    return np.argmax(probas)

#Step5:Loop of generating text
for epoch in range(1,60):                        #train the model 60 epochs
    print('\nepoch',epoch)
    model.fit(x,y,batch_size=128,epochs=1)
    start_index = random.randint(0,len(text)-max_len-1)
    generated_text = text[start_index:start_index+max_len]
    print('\n--- Generating with seed:'+generated_text+'"')
    for temperature in [0.2,0.5,1.0,1.2]:
        print('\n--- Temperature:',temperature)
        sys.stdout.write(generated_text)
        for i in range(400):                     #generate 400 chars
            sampled =  np.zeros((1,max_len,len(chars)))
            for t,char in enumerate(generated_text):
                sampled[0,t,char_indices.get(char)]=1
            preds = model.predict(sampled,verbose=0)[0]      #return a list,must use index to get the element
            next_index = Sample(preds,temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]              #move back a char

            sys.stdout.write(next_char)