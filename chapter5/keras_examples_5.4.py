"""
Visualization of Convolution Net
1,Visualize the middle output of the CNN
2,Visualize the filter of the CNN
3,Visualize the class activation by heat map of CNN
"""
import  numpy as np
import  matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import  image
from keras import  models
from keras.applications import VGG16
from keras import backend as K

model = load_model('cat_and_dogs_small_1.h5')    #load model
model.summary()     #see the structure of the model

img_path = '/home/zhangjc/PycharmProjects/Deep-Learning-with-Python/dataset/cats_and_dogs_samll/test/cats/cat.1700.jpg'
img = image.load_img(img_path,target_size=(150,150))
img_tensor = image.img_to_array(img)

#Reshape the img_tensor
#way-1
img_tensor = img_tensor.reshape((1,)+img_tensor.shape)
#way-2
#img_tensor = np.expand_dims(img_tensor,axis= 0)  #add a dims in the axis 0
img_tensor /= 255.    #now the shape of img_tensor is (1,150,150,3),you can show the picture by plt.imshow(img_tensor[0])

"""==================================Part1:Visualize the output of the middle layers==================================="""
# #Step1:Build a model
# layers_outputs = [layer.output for layer in model.layers[:8]]
# activation_model = models.Model(inputs = model.input,outputs = layers_outputs)
# activations = activation_model.predict(img_tensor)  #the shape of the activations is (8,1,148(...),148(...),filters)
#
# #here show the picture must use plt.imshow(activation[0][0,:,:,3])
#
# #Step2:Visualize all filter of all middle layers
# layer_names = []
# for layer in model.layers[:8]:
#     layer_names.append(layer.name)
#
# image_per_row = 16
#
# for layer_name,layer_activation in zip(layer_names,activations):
#     n_features = layer_activation.shape[-1]     #number of the filters
#     #shape of layer_acvtivation :(1,size,size,filters)
#     size = layer_activation.shape[1]
#
#     n_rows = n_features//image_per_row
#     display_grid = np.zeros((n_rows*size,image_per_row*size))  #
#     for row in range(n_rows):
#         for col in range(image_per_row):
#             channel_image = layer_activation[0,:,:,row*image_per_row+col]
#             #beautify the image data
#             channel_image -= channel_image.mean()
#             channel_image /= (channel_image.std()+1e-5)   #avoid to divide by 0
#             channel_image *=64
#             channel_image +=128
#             channel_image = np.clip(channel_image,0,255).astype('uint8')
#
#             display_grid[row * size:(row + 1) * size, col * size:(col + 1) * size] = channel_image
#     #must rescale the figure,or the picture can't be shown
#     scale = 1./255
#     plt.figure(figsize=(scale*display_grid.shape[1],scale*display_grid.shape[0]))
#
#     plt.title(layer_name)
#     plt.grid(False)
#     plt.imshow(display_grid,aspect='auto',cmap = 'viridis')


"""===================================Part2:Visualize the filters model======================================="""
"""
A convolution net is to train a group of filters,and then divide the image into differents filters.
# """
model  = VGG16(weights = 'imagenet',include_top = False)
layer_name ='block3_conv1'
filter_index = 0

def deprocess_image(x):
    """
    param x:
    return:A image array can be shown
    """
    x -= x.mean()
    x /= (x.std()+1e-5)
    x*=0.1
    x+=0.5
    x = np.clip(x,0,1)

    x*= 255
    x = np.clip(x,0,255).astype('uint8')
    return x

def generate_pattern(layer_name,filter_index,size = 150):
    #build a loss function to maxsize the output of filter_index filter
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])

    # PA:here gradient function is not the same with the np.gradient,it return a list consist of some tensors
    # PA:mean(),square(),Sqrt() is also not the same with that in np
    #calculate the gradients,grads is a tensor
    grads =K.gradients(loss,model.input) [0]       #must after we have got the loss
    #normalize the gradients
    grads /=(K.sqrt(K.mean(K.square(grads)))+1e-5)
    #return the loss and gradients of input image
    iterate = K.function([model.input],[loss,grads])
    #create the initial image data
    input_img_data = np.random.random((1,size,size,3))*20+128

    #use the gradient ascent to get the special image
    step =1
    for i in range(40):
        loss_value,grads_value = iterate([input_img_data])
        input_img_data += grads_value*step

    img = input_img_data[0]
    img = deprocess_image(img) #convert the data-type to uint8 to show
    return img

#show the all filters model of a layers
layer_name = 'block2_conv1'
size = 64         #every grid pixel
margin = 5        #gap pixel

results = np.zeros((8*size+7*margin,8*size+7*margin,3))
#results = np.zeros((8*size,8*size,3))
for k in range(1,5):
    layer_name = 'block'+str(k)+'_conv1'
    for i in range(8):
        for j in range(8):
            filter_img = generate_pattern(layer_name,i*8+j,size = size)
           # results[i*size:(i+1)*size,j*size:(j+1)*size,:] = filter_img

            horizontal_start = i*size+i*margin
            horizontal_end = horizontal_start+size
            vertical_start = j*size+j*margin
            vertical_end = vertical_start+size
            results[horizontal_start:horizontal_end,vertical_start:vertical_end,:] = filter_img

    if results.dtype!="uint8":
        print('The image data isn\'t the uint8')
        results = results.astype("uint8")
    plt.figure(figsize = (20,20 ))
    plt.savefig(layer_name+'.png')
    plt.imshow(results)


"""=========================================Visualize the Class Activation Map==========================================="""
"""
It's a way to let's know what part of the image is the most important in the image classfication,it can show the elements's
importance of the image by heap map.
"""