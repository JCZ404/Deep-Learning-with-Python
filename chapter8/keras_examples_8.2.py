"""
利用Keras实现DeepDream，生成迷幻图片
张量：用Keras后端创建的，用于张量运算的数据结构，一个np数组不是张量。 K.is_tensor
keras张量：由Keras的layers输出的，或Input返回的是Keras张量  K.is_keras_tensor
K.shape:返回张量或者变量的符号尺寸，本事是一个张量。要想获得元祖形式的形状，应该使用int_shape
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import imageio
from keras.applications import inception_v3
from keras.preprocessing import image            #image processing module
from keras import backend as K

K.set_learning_phase(0)   #set trainable False

model = inception_v3.InceptionV3(weights = "imagenet",include_top = False)

#Set Deepdream Mode
layer_contributions = {'mixed2':0.2,'mixed3':3.0,'mixed4':2.0,'mixed5':1.5}

layer_dict = dict([(layer.name,layer)for layer in model.layers])

dream = model.input       #save the image

#Define Loss need to maximize
loss = K.variable(0.)  #add the contribution of layers into the scalar when define loss
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output

    scaling = K.prod(K.cast(K.shape(activation),'float32'))   #scaling is a tensor scalar
    loss += coeff*K.sum(K.square(activation[:,2:-2,2:-2,:]))/scaling

#Define Gradients
grads = K.gradients(loss,dream)[0]  #calculate the gradients related to dream(input image)
grads /= K.maximum(K.mean(K.abs(grads)),1e-7)  #gradients normalization

#Define Outputs
outputs = [loss,grads]


fetch_loss_and_grads = K.function([dream],outputs)  #define a function to get the loss and grads of a input image

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_value = outs[1]
    return loss_value,grad_value

def gradient_ascent(x,iterations,step,max_loss =None):
    for i in range(iterations):                         #run iteration times gradients_ascent
        loss_value,grad_value = eval_loss_and_grads(x)
        if max_loss is not None and loss_value>max_loss:
            break
        print('...Loss value at',i,':',loss_value)
        x += step*grad_value
    return x

#Some processing function
def preprocess_image(image_path):
    #transform a image into the tensor that inception_v3 can deal with
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def deprocess_image(x):
    #transform a tensor into a true image
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,float(size[0]) / img.shape[1],float(size[1]) / img.shape[2],1)
    return scipy.ndimage.zoom(img, factors, order=1)

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    imageio.imsave(fname, pil_img)

step = 0.01
num_octave = 3        #num of scale running gradient ascent
octave_scale = 1.4
iterations = 20

max_loss = 10

base_image_path = "/home/zhangjc/PycharmProjects/test/test.jpg"
img = preprocess_image(base_image_path)   #load iamge into a numpy array


original_shape = img.shape[1:3]
successive_shapes = [original_shape]

#Define the different scale of image runnnig gradient ascent later
#successive_shapes consist of different shape tupletensorflow.python.framework.errors_impl.InternalError: Failed to create session.
for i in range(1,num_octave):
    shape = tuple([int(dim/(octave_scale**i))for dim in original_shape])
    successive_shapes.append(shape)
print("==================",successive_shapes,"===================")
successive_shapes = successive_shapes[::-1]  #reverse the tuple,become ascending order
print("==================",successive_shapes,"===================")

original_img = np.copy(img)
shrunk_original_img = resize_img(img,successive_shapes[0])   #put the image into the smallest size
print("==================",shrunk_original_img.shape,"===================")


for shape in successive_shapes:
    print('Processing the image shape:',shape)
    img = resize_img(img,shape)           #upscale the image
    img = gradient_ascent(img,iterations = iterations,step = step,max_loss = max_loss)

    upscaled_shrunk_original_image = resize_img(shrunk_original_img,shape)   #upscale the shrunk_original_iamge,it will loss some information
    same_size_original = resize_img(original_img,shape)
    lost_detail = same_size_original-upscaled_shrunk_original_image

    img += lost_detail      #add the loss details into the original image
    shrunk_original_img = resize_img(original_img,shape)
    save_img(img,fname='dream_at_scale_'+str(shape)+'.png')
save_img(img,fname='final_dream.png')





