"""
Keras实现神经风格迁移，将一张原始的图片，转换成一张具有另一张绘画或图片风格的图片。
关键是内容与风格的定义
"""
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import time
from keras.preprocessing import image
from keras.applications import vgg19        #VGG19 is simpler than VGG16
from scipy.optimize import fmin_l_bfgs_b
from imageio import imsave


target_image_path = "/home/zhangjc/PycharmProjects/test/test.jpg"
style_reference_image_path = "/home/zhangjc/PycharmProjects/test/test.jpg"

width,height = image.load_img(target_image_path).size
img_height = 400
img_width = int(width*(img_height/height))      #resize the image

def preprocess_image(image_path):
    img = image.load_img(image_path,target_size=(img_height,img_width))      #must resize the image into the same size
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    img = vgg19.preprocess_input(img)           #actually is doing some normalization work
    return img

def deprocess_image(x):
    #the reverse preprocess_input
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    x = x[:,:,::-1]   #BGR to RGB

    x = np.clip(x,0,255).astype('uint8')
    return x

#Build VGG19 model and apply it to the three images
#all of the image data must be transformed to the tensor
target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))
combination_image = K.placeholder((1,img_height,img_width,3))

input_tensor = K.concatenate([target_image,style_reference_image,combination_image],axis=0)

model = vgg19.VGG19(weights = 'imagenet',include_top = False,input_tensor = input_tensor)

#Define the loss of content
def content_loss(base,combination):
    return K.sum(K.shape(base-combination))

#Define the loss of style
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x,(2,0,1)))
    gram = K.dot(features,K.transpose(features))
    return gram

def style_loss(style,combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_width*img_height
    return K.sum(K.square(S-C))/(4*(size**2)*(channels**2))

#Define the loss of total variation,avoiding the image to a pixel
def total_variation_loss(x):
    a = K.square(x[:,:img_height-1,:img_width-1,:]-x[:,1:,:img_width-1,:])
    b = K.square(x[:,:img_height-1,:img_width-1,:]-x[:,:img_height-1,1:,:])
    return K.sum(K.pow((a+b),1.25))

#Define the final loss
output_dict = dict([(layer.name,layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layer = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

loss = K.variable(0)         #define total loss,attention,it't a tensor
layer_features = output_dict[content_layer]     #the output is [target_image,reference_image,combinatoin_image]
target_image_features = layer_features[0,:,:,:]
combination_features = layer_features[2,:,:,:]
loss += content_weight*content_loss(target_image_features,combination_features)

for layer_name in style_layer:
    layer_features = output_dict[layer_name]
    style_reference_features = layer_features[1,:,:,:]
    combination_features = layer_features[2,:,:,:]
    s1 = style_loss(style_reference_features,combination_features)
    loss += (style_weight/len(style_layer))*s1

loss += total_variation_loss(combination_image)*total_variation_weight   #is the combination image exist so far?

#Define the gradients decent
grads = K.gradients(loss,combination_image)[0]     #attension the index

fetch_loss_and_grads = K.function([combination_image],[loss,grads]) #input and output must be a list

class Evaluator(object):       #wrap the fetch_loss_and_grads so that we can use the loss and grads function apart
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self,x):
        assert self.loss_value is None
        x = x.reshape((1,img_height,img_width,3))
        outs = fetch_loss_and_grads([x])   #x must be 4-dims
        loss = outs[0]
        grads = outs[1].flatten().astype('float64')
        self.loss_value = loss
        self.grads_values = grads
        return self.loss_value

    def grads(self,x):
        assert self.grads_values is None
        grads_values = np.copy(self.grads_values)  #reflush the loss_value and grads_value
        self.loss_value = None
        self.grads_values = None
        return grads_values

evaluator = Evaluator()

#Loop of gradients decent


# result_prefix = 'my_result'
# iterations = 20
# x = preprocess_image(target_image_path)          #load original image
# x = x.flatten()
# for i in range(iterations):
#     print("Start of iteration:",i)
#     start_time = time.time()
#     x,min_val,info = fmin_l_bfgs_b(evaluator.loss,x,fprime=evaluator.grads,maxfun=20)  #
#     print("Current loss value:",min_val)
#     img = x.copy().reshape((img_height,img_width,3))
#     img = deprocess_image(img)
#     fname = result_prefix+'_at_iteration_%d.png'%i
#     imsave(fname,img)
#     print('Image save as ',fname)
#     end_time = time.time()
#     print('Iteration %d completed in %ds'%(i,end_time-start_time))
#
