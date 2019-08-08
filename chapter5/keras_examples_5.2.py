import os,shutil
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import layers
from keras import models
from keras import optimizers
#


original_dataset_dir = '/home/zhangjc/PycharmProjects/Deep-Learning-with-Python/dataset/train/'
#最好使用Linux系统的习惯，文件目录用斜杆/隔开，上面要改为 'F:/Pycharm_projects/Test/dataset/train/',最后要用斜杆结尾
#这样在使用os模块时join方法才能正常工作，因为没有/，os默认用/连接


base_dir = '/home/zhangjc/PycharmProjects/Deep-Learning-with-Python/dataset/cats_and_dogs_samll'      #保存较小数据的目录
if not os.path.exists(base_dir):
    os.mkdir(base_dir)           #创建路径

# 在存放较小数据集的路径下生成存放训练数据,验证数据,测试数据的路径
train_dir = os.path.join(base_dir,'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
test_dir = os.path.join(base_dir,'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)


train_cats_dir = os.path.join(train_dir,'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir,'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir,'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir,'dogs')
if not os.path.exists(validation_dogs_dir):
    os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir,'cats')
if not os.path.exists(test_cats_dir):
    os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir,'dogs')
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)

#将数据复制到相应的目录下
#猫
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_cats_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

#狗
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_dogs_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src,dst)

#构建模型
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation= 'relu',input_shape=(150,150,3)) )   #input tensor:(image_height,image_width,image_channels)
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())                          #make the 3D tensor flatten to 1D
model.add(layers.Dropout(0.5))                       #using dropoupt to reduce the overfiting,it should be added before the dense
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


#编译模型
model.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics=['acc'])


#使用数据增强，利用生成器从目录中读取图像,注意是增强训练数据
train_datagen = ImageDataGenerator(
    rescale= 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)                                                          #将训练数据进行数据增强
test_datagen = ImageDataGenerator(rescale=1./255)          #将图像进行缩放，测试数据不能进行数据增强


train_generator = train_datagen.flow_from_directory(train_dir,              #目标样本所在目录
                                                    target_size=(150,150),  #将图像大小调整为150*150
                                                    batch_size=20,          #每批数据有20个样本
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                  target_size=(150,150),
                                                  batch_size=20,
                                                  class_mode = 'binary')

history = model.fit_generator(train_generator,                             #因为是用了生成器来生成训练数据，所以使用生成器版的拟合
                              steps_per_epoch=100,                         #每一轮训练使用100个批次的数据
                              epochs=100,                                   #训练30轮
                              validation_data=validation_generator,        #验证数据可以是生成器，也可以是数据元组
                              validation_steps=50)

model.save('cat_and_dogs_small_1.h5')

history_dict = history.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1,len(acc)+1)

plt.figure(1)
plt.plot(epochs,acc,'bo',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure(2)
plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title("Training and Validation loss")
plt.legend()

plt.show()


