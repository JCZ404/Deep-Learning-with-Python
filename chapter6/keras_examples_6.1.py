"""
Processing the text sequence.
1,Text vectorization
"""
import numpy as np
import string
from  keras.preprocessing.text import  Tokenizer

#Word level vectorization---one-hot

samples = ['The cat sat on the mat.','The dog ate my homework.']
# token_index = {}      #build a dict:word:index
# for sample in samples:
#     for word in list(sample.split()):
#         if word  not in token_index.keys():
#             token_index[word] = len(token_index)+1          #because we don't use the index 0
#
# max_length = 10       #only consider the top 10 words
# #Build the ont-hot tensor
# results_word = np.zeros((len(samples),max_length,max(token_index.values())+1))
# for i in range(len(samples)):
#     for j,word in list(enumerate(samples[i].split())):       #do't need to use three loop
#         index = token_index.get(word)
#         results_word[i,j,index] =1


#Char level vectorization

# characters = string.printable                                #get all printable ASCII char
# token_index = dict(zip((1,len(characters)+1),characters))
# max_length = 50
# results_char = np.zeros((len(samples),max_length,max(token_index.keys())))
# for i,sample in enumerate(samples):
#     for j,char in enumerate(sample):                         #enumerate(string) --->[(index,char)...]
#         index= token_index.get(char)
#         results_char[i,j,index]=1
#

#Use keras to realize the word-level one-hot
tokenizer = Tokenizer(num_words=1000)                        #only consider the top 1000 common words
tokenizer.fit_on_texts(samples)                              #build word index

sequence = tokenizer.texts_to_sequences(samples)             #convert string into the list of index
one_hot_results = tokenizer.texts_to_matrix(samples)         #convert string into one-hot matrix

word_index = tokenizer.word_index                            #get word index



