#using pretrained Convolutional neural network,such as :VGG,ResNet,Inception,Inception-ResNet,Xception
#there are two ways to use the pretrained network:feature extraction,fine-tuning



import os
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.preprocessing.image import  ImageDataGenerator
from keras import models
from keras import layers
from keras import  optimizers
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement = True
# sess= tf.Session(config = config)

#define a function to make the smooth curve by Exponential moving average
def smooth_curve(points,factor = 0.8):
    smooth_points =[]
    for point in points:
        if smooth_points:
            previous = smooth_points[-1]
            smooth_points.append(previous*factor+(1-factor)*points)
        else:
            smooth_points.append(points)
    return smooth_points

#
# def get_session(gpu_fraction=0.3):
#     """
#     This function is to allocate GPU memory a specific fraction
#     Assume that you have 6GB of GPU memory and want to allocate ~2GB
#     """
#     num_threads = os.environ.get('OMP_NUM_THREADS')
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#
#     if num_threads:
#         return tf.Session(config=tf.ConfigProto(
#             gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
#     else:
#         return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
# KTF.set_session(get_session(0.5))  # using 40% of total GPU Memory


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"   #使用GPU

#Initialize the VGG16 model
conv_base = VGG16(weights = 'imagenet',include_top = False,input_shape=  (150,150,3))
base_dir = '/home/zhangjc/PycharmProjects/Deep-Learning-with-Python/dataset/cats_and_dogs_samll'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')


""""=======================Part-1:Feature Extraction----Reuse the Convolution Base==============================="""


#Way-1:Don't use Data Enhance
#Step1:Prepare data---don't use the data enhence
# datagen = ImageDataGenerator(rescale=1./255)            #using generator to load data
# batch_size = 20


#Step2:Get the features and store them into a numpy array
# def extract_features(directory,sample_count):
#     features= np.zeros(shape=(sample_count,4,4,512))           #The shape of feature is (4,4,512)
#     labels = np.zeros(shape=(sample_count))
#     generator = datagen.flow_from_directory(directory,target_size=(150,150),batch_size=batch_size,class_mode='binary')
#
#     i = 0
#     for inputs_batch,labels_batch in generator:
#         features_batch = conv_base.predict(inputs_batch)         #Get the features from the conv_base of VGG model by using predict
#         features[i*batch_size:(i+1)*batch_size] = features_batch
#         labels[i*batch_size:(i+1)*batch_size] = labels_batch
#         i += 1
#         if i*batch_size >= sample_count:
#             break
#     return features,labels
#
# train_features,train_labels = extract_features(train_dir,2000)
# validation_features,validation_labels = extract_features(validation_dir,1000)
# test_features,test_labels = extract_features(test_dir,1000)
#
# #Step3:Flatten the features in order to input them to the dense net
# train_features = np.reshape(train_features,(2000,4*4*512))
# validation_features = np.reshape(validation_features,(1000,4*4*512))
# test_features = np.reshape(test_features,(1000,4*4*512))
#
##Step4:Define model
#
# model = models.Sequential()
# model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1,activation='sigmoid'))   #only two category
#
#
# #Step5:Compile model
# model.compile(optimizer = optimizers.RMSprop(lr = 2e-5),loss = 'binary_crossentropy',metrics=['acc'])
#
# #Step6:Train model
# history = model.fit(train_features,train_labels,epochs=30,batch_size=20,validation_data = (validation_features,validation_labels))
#
# #Step7:Visualize the results
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss= history.history['val_loss']
#
# epochs = range(1,len(acc)+1)       #it will calculate the acc and loss afer a epoch finished
#
# plt.figure(1)
# plt.plot(epochs,acc,'bo',label = 'Training acc')
# plt.plot(epochs,val_acc,'b',label = 'Validation acc')
# plt.title('Training and Validation accuracy')
# plt.legend()
#
# plt.figure(2)
# plt.plot(epochs,loss,'bo',label = 'Training loss')
# plt.plot(epochs,val_loss,'b',label = 'Validation loss')
# plt.title('Training and Validation loss')
# plt.legend()
# plt.show()
#

#Way-2:Use Data Enhence---train the model directly

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#Step1:Freeze the conv_base in orde to avoid the change of the weights of conv_base
#see the number of trainble weigths tensor
print('Total number of trainable weights tensor before freeze the conv_base:',len(model.trainable_weights))
#freeze the conv_base
conv_base.trainable= False
print('Total number of trainable weights tensor after freeze the conv_base:',len(model.trainable_weights))

#Step2:Prepare the data---use data enhence
train_datagen = ImageDataGenerator(
    rescale= 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_dategen = ImageDataGenerator(rescale=1./255)    #validation data can't be enhenced

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode= 'binary')      #because we use the loss of binary_crossentropy when compile model
validation_generator = test_dategen.flow_from_directory(validation_dir,
                                                        target_size=(150,150),
                                                        batch_size=20,
                                                        class_mode='binary')

#Step3:Compile model
model.compile(optimizer = optimizers.RMSprop(lr = 2e-5),loss = 'binary_crossentropy',metrics=['acc'])

#Step4:Train model---use fit_generator
#here steps_per_epoch means the number of batch in a epoch,also means run steps_per_epoch times gradient decent of a epoch
history  = model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data = validation_generator,validation_steps=50)

#Step5:Visualize the results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss= history.history['val_loss']

epochs = range(1,len(acc)+1)       #it will calculate the acc and loss afer a epoch finished

plt.figure(1)
plt.plot(epochs,acc,'bo',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure(2)
plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

""""======================================Part-2:Fine-tuning the model====================================="""
"""
Fine-tunning the model must be done after the new dense net we added has been trained,or the weights of the base network
will be destroyed
 """
# #Step1:Freeze the layers of conv_base
# conv_base.trainable = True
#
# set_trainable = False
# for layer in conv_base.layers:         #unfreeze the layers after the block5_conv1
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False
#
# #Step2:Compile model
# model.compile(loss = 'binary_crossentropy',optimizer = optimizers.RMSprop(1e-5),metrics=['acc'])
# history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data = validation_generator,validation_steps=50)
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss= history.history['val_loss']
#
# epochs = range(1,len(acc)+1)       #it will calculate the acc and loss afer a epoch finished
# #plot the smooth curve
# plt.figure(1)
# plt.plot(epochs,smooth_curve(acc),'bo',label = 'Training acc')
# plt.plot(epochs,smooth_curve(val_acc),'b',label = 'Validation acc')
# plt.title('Training and Validation accuracy')
# plt.legend()
#
# plt.figure(2)
# plt.plot(epochs,smooth_curve(loss),'bo',label = 'Training loss')
# plt.plot(epochs,smooth_curve(val_loss),'b',label = 'Validation loss')
# plt.title('Training and Validation loss')
# plt.legend()
# plt.show()
#
#
# #Step6:Test Model
# test_generator = test_dategen.flow_from_directory(test_dir,target_size=(150,150),class_mode='binary',batch_size=20)
#
# test_loss,test_acc = model.evaluate_generator(test_generator,steps=50)
# print(test_loss,test_acc)