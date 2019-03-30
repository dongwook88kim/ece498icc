import tensorflow as tf
#from tensorflow.keras import layers
from tensorflow import keras
from PIL import Image
print(tf.VERSION)
print(tf.keras.__version__)
import numpy as np
import struct

def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

'''
Weights_list is the parameter sets of the networks

It's structure is like:
[
[],
[layer a's weights, layer a's bias,...],
[layer b's weights]
]
'''
import pickle
with open ('./params_sets', 'rb') as fp:
    weights_list = pickle.load(fp)
weights_list.pop(0)

#for i in range(len(weights_list)):
#    print(len(weights_list[i]), len(weights_list[i][0])) 


#this is the model for the testing part
#model = tf.keras.models.Sequential(
model = keras.Sequential(
    [
    #first dw module
    keras.layers.DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=False),
    keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5,trainable=False), #1
    keras.layers.ReLU(4.0), #2
    keras.layers.Conv2D(48,(1,1), padding='same',use_bias=False,strides=(1, 1)),
    keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5,trainable=False), #4
    keras.layers.ReLU(4.0), #5
    #maxpooling
    keras.layers.MaxPool2D(strides =(2,2)), #6
    #second dw module
    keras.layers.DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=False),
    keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5,trainable=False), #8
    keras.layers.ReLU(4.0), #9
    keras.layers.Conv2D(96,(1,1), padding='same',use_bias=False,strides=(1, 1)),
    keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5,trainable=False), #11
    keras.layers.ReLU(4.0), #12
    #maxpooling
    keras.layers.MaxPool2D(strides =(2,2)), #13
    #third dw module
    keras.layers.DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=False),
    keras.layers.BatchNormalization(momentum=0.1,epsilon=1e-5,trainable=False), #15
    keras.layers.ReLU(4.0), #16
    keras.layers.Conv2D(192,(1,1), padding='same',use_bias=False,strides=(1, 1)),
    keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5,trainable=False), #18
    keras.layers.ReLU(4.0), #19
    #maxpooling
    keras.layers.MaxPool2D(strides =(2,2)), #20
    #fourth dw module
    keras.layers.DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=False),
    keras.layers.BatchNormalization(momentum=0.1,epsilon=1e-5,trainable=False), #22
    keras.layers.ReLU(4.0), #23
    keras.layers.Conv2D(384,(1,1), padding='same',use_bias=False,strides=(1, 1)),
    keras.layers.BatchNormalization(momentum=0.1,epsilon=1e-5,trainable=False), #25
    keras.layers.ReLU(4.0), #26
    #fifth dw module
    keras.layers.DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=False),
    keras.layers.BatchNormalization(momentum=0.1,epsilon=1e-5,trainable=False), #28
    keras.layers.ReLU(4.0), #29
    keras.layers.Conv2D(512,(1,1), padding='same',use_bias=False,strides=(1, 1)),
    keras.layers.BatchNormalization(momentum=0.1,epsilon=1e-5,trainable=False), #31
    keras.layers.ReLU(4.0), #32
    #output
    keras.layers.Conv2D(10,(1,1), padding='same',use_bias=False,strides=(1, 1)),
    ]
)

model.build(input_shape=(1,320,160,3))
model.summary()

model.trainable = False

#### Set weights by layers
print("Setting weights")

# DepthwiseConv2D (1, 3, 3, 3, 1) -> (# of data, conv_x, conv_y, input channel(3), 1)  /// in: 3ch, out: 3ch
# batch_norm (4, 3) -> ( (gamma, beta, mean, stdv), output channels(3) ) /// out: 3ch
# Conv2D (1, 1, 1, 3, 48) -> (# of data, conv_x, conv_y, input channels(3), output channels(48) ) /// in: 3ch, out: 48ch
# batch_norm (4, 48) ( (gbms), output channels(48) ) /// out: 48ch

## Function takes inputs (# data, dim_x , dim_y, in chs). 
##                conv filter (# of data, conv_x, conv_y, in chs, out chs)
##                outputs (# of data, dim_x-conv_x, dim_y-conv_y, in_chs*out chs)
##                batch_norm ( factors, in_chs*out_chs)
##                outputs (# of data, dim_x-conv_x, dim_y,conv_y, in_chs*out chs)


layer_to_ignore = [2,5,6,9,12,13,16,19,20,23,26,29,32]
j=0

for i in range(34):
    if i not in layer_to_ignore:
        print(np.array(weights_list[j]).shape)
        #model.layers[i].set_weights(np.array(weights_list[j]))
        model.layers[i].set_weights( weights_list[j] )
        j += 1


#model.set_weights(weights_list)

def load_input(path):
    img = Image.open(path)
    img = img.resize((320,160))
    input_img = np.asarray(img).astype(np.float32)
    input_img = (input_img/255 - 0.5)/0.25
    return input_img[np.newaxis,:]

'''
This is the function to get the predict box (x,y,w,h)
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def get_box(output):
    anchors = [1.4940052559648322, 2.3598481287086823, 4.0113013115312155, 5.760873975661669]
    h = output.shape[2]
    w = output.shape[3]
    output = output.reshape(2,5,800).transpose(1,0,2).flatten().reshape(5,1600)
    grid_x = np.tile(np.tile(np.linspace(0,w-1,w),h).reshape(h,w),(2,1,1)).flatten()
    grid_y =np.tile(np.tile(np.linspace(0,h-1,h),w).reshape(w,h).T,(2,1,1)).flatten()
    xs = sigmoid(output[0]) + grid_x
    ys = sigmoid(output[1]) + grid_y
    anchor_w = np.zeros(1600)
    anchor_h = np.zeros(1600)
    anchor_w[0:800] = anchors[0]
    anchor_w[800:1600] = anchors[2]
    anchor_h[0:800] = anchors[1]
    anchor_h[800:1600] = anchors[3]
    ws = np.exp(output[2]) * anchor_w
    hs = np.exp(output[3]) * anchor_h
    ind = np.argmax(output[4])
    bcx = xs[ind]
    bcy = ys[ind]
    bw = ws[ind]
    bh = hs[ind]
    box = [bcx/w, bcy/h, bw/w, bh/h]
    return box

'''
This is the cell to test your weights correctness.

The output should be :
[0.8880645155906677, 0.6772263944149017, 0.02124013871572325, 0.058586649582813566]
'''
input_img = load_input('./images/2.jpg')
output = model.predict(input_img).transpose(0,3,1,2)
print (get_box(output))
#print(model.get_weights())
'''
Now finish the function to compute the iou between two given box.

You can refer to the website: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

'''

def bbox_iou(box1, box2):
    '''your code here'''
    # (x,y,w,h) to (x_large, y_large, x_small, y_small)
    # x_large = x + w/2
    # x_small = x - w/2
    # y_large = y + h/2
    # y_small = y - h/2

    #

    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = min(box1[0]+box1[2]/2, box2[0]+box2[2]/2)
    y1 = min(box1[1]+box1[3]/2, box2[1]+box2[3]/2)
    x2 = max(box1[0]-box1[2]/2, box2[0]-box2[2]/2)
    y2 = max(box1[1]-box1[3]/2, box2[1]-box2[3]/2)

    # compute the area of intersection rectangle
    interArea = max(0, x1 - x2) * max(0, y1 - y2)

    # compute the area of both the prediction and ground-truth rectangles
    box1Area = box1[2] * box1[3]
    box2Area = box2[2] * box2[3]

    # compute the intersection over union by taking the intersection area
    # and dividing it by the sum of prediction + ground-truth areas - the interesection area
    iou = interArea / (box1Area + box2Area - interArea)
    print(iou)

    # return the intersection over union value
    return iou

import json
with open('groundtruth.txt', 'r') as outfile:
    lines = json.load(outfile)

'''
The iou should be about 67%
'''
#avg_iou = 0
#how_many = 0
#for line in lines:
#    how_many += 1
#    if (how_many%100==0):
#        print(how_many)
#    input_img = load_input(line[0])
#    output = model.predict(input_img).transpose(0,3,1,2)
#    avg_iou+= bbox_iou(get_box(output),line[1])
#avg_iou = avg_iou/1000
#print (avg_iou)

#model_no_bn = tf.keras.models.Sequential(
model_no_bn = keras.Sequential(
    [
    #first dw module
    keras.layers.DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=True),
    #keras.layers.Conv2D(3,(3, 3),padding='same',strides=(1,1),use_bias=True),
    keras.layers.ReLU(4.0), #1
    keras.layers.Conv2D(48,(1,1), padding='same',use_bias=True,strides=(1, 1)),
    keras.layers.ReLU(4.0), #3
    #maxpooling
    keras.layers.MaxPool2D(strides =(2,2)), #4
    #second dw module
    keras.layers.DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=True),
    #keras.layers.Conv2D(48,(3, 3),padding='same',strides=(1,1),use_bias=True),
    keras.layers.ReLU(4.0), #6
    keras.layers.Conv2D(96,(1,1), padding='same',use_bias=True,strides=(1, 1)),
    keras.layers.ReLU(4.0), #8
    #maxpooling
    keras.layers.MaxPool2D(strides =(2,2)), #9
    #third dw module
    keras.layers.DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=True),
    #keras.layers.Conv2D(96,(3, 3),padding='same',strides=(1,1),use_bias=True),
    keras.layers.ReLU(4.0), #11
    keras.layers.Conv2D(192,(1,1), padding='same',use_bias=True,strides=(1, 1)),
    keras.layers.ReLU(4.0), #13
    #maxpooling
    keras.layers.MaxPool2D(strides =(2,2)), #14
    #fourth dw module
    keras.layers.DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=True),
    #keras.layers.Conv2D(192,(3, 3),padding='same',strides=(1,1),use_bias=True),
    keras.layers.ReLU(4.0), #16
    keras.layers.Conv2D(384,(1,1), padding='same',use_bias=True,strides=(1, 1)),
    keras.layers.ReLU(4.0), #18
    #fifth dw module
    keras.layers.DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=(1,1),use_bias=True),
    #keras.layers.Conv2D(384,(3, 3),padding='same',strides=(1,1),use_bias=True),
    keras.layers.ReLU(4.0), #20
    keras.layers.Conv2D(512,(1,1), padding='same',use_bias=True,strides=(1, 1)),
    keras.layers.ReLU(4.0), #22
    keras.layers.Conv2D(10,(1,1), padding='same',use_bias=False,strides=(1, 1)), #23
    ]
)

model_no_bn.build(input_shape=(1,320,160,3))
model_no_bn.summary()

model_no_bn.trainable = False


## Let's get weights shape
print("Let's get weights!")
print(np.array(model_no_bn.get_weights()[0]).shape)
print(np.array(model_no_bn.get_weights()[1]).shape)
print(np.array(model_no_bn.get_weights()[2]).shape)
print(np.array(model_no_bn.get_weights()[3]).shape)

'''
Write down the code to absorb bn layer into conv layer and maintain the same output as the original model. (please refer to HW2 Q4)
'''
def fold_batch_norm_depthwise(conv_layer, bn_layer):
    """Fold the batch normalization parameters into the weights for 
       the previous layer."""
    conv_weights = conv_layer.get_weights()
    #print("conv_weight shape = ", np.array(conv_weights).shape)

    ## depthwise conv layer has 5 dim. Ex) (1,3,3,3,1) = (batches, conv_x, conv_y, input channels, 1)
    ## batch norm layer has 2 dim. Ex) (4, 3) = (batch params, output channels)
    ## output layer has 5 dim. Ex) (1,3,3,3,1) = (batches, conv_x, conv_y, output channels, 1)
    
    # Keras stores the learnable weights for a BatchNormalization layer
    # as four separate arrays:
    #   0 = gamma (if scale == True)
    #   1 = beta (if center == True)
    #   2 = moving mean
    #   3 = moving variancei
    bn_weights = bn_layer.get_weights()
    #print("bn_weight shape = ", np.array(bn_weights).shape)
    #print("bn_weight gamma = ", np.array(bn_weights[0]))

    gamma = bn_weights[0]
    beta = bn_weights[1]
    mean = bn_weights[2]
    variance = bn_weights[3]

    # Now, batch params have # of output channels. Ex) 3

    epsilon = 1e-5
    
    #new_weights = np.array(conv_weights) * gamma / np.sqrt(variance + epsilon)
    new_weights = np.array(conv_weights)
    
    for i in range(len(bn_weights[0])):
        new_weights[:,:,:,i,:] = np.array(conv_weights)[:,:,:,i,:] * gamma[i] / np.sqrt(variance[i] + epsilon)

    new_bias = beta - mean * gamma / np.sqrt(variance + epsilon)

    # Prints for test
    #print("conv_seiths dimension", np.array(conv_weights).shape)
    #print("new_weights dimension", new_weights.shape)
    #print("mean dimension", np.mean(new_weights, axis=4).shape)
    #print("resape", np.mean(new_weights, axis=4)[..., None].shape)
    #new_weights_2 = np.mean(new_weights, axis=4)[..., None]

    # Check upper bound
    up_bound = np.max( [abs(np.max(new_weights)), abs(np.min(new_weights))] )
    print("Depthwise up_bound = ", up_bound)
    if up_bound <= 2:
        e = 0
    else:
        e = np.ceil ( np.log2 ( np.ceil( np.log2(up_bound/2) ) ) ) + 1
    print("s, e, m = ", 1, e, 8-1-e)

    up_bound_binary = float_to_bin(up_bound)
    print("Binary representation of up_baound = ", up_bound_binary )
    print("its type", type(up_bound_binary))

    # See maximum and minimum weights
    #print("(float32) Maximum new weights = ", np.max(new_weights))
    #print("(float32) Minimum new weights = ", np.min(new_weights))

    # Change float32 to float16 to float32 to drop precisions
    #new_weights = np.float16(new_weights)
    #new_weights = np.float32(new_weights)

    # See maximum and minimum weights
    #print("(float16) Maximum new weights = ", np.max(new_weights))
    #print("(float16) Minimum new weights = ", np.min(new_weights))



    return new_weights, new_bias

def fold_batch_norm_pointwise(conv_layer, bn_layer):
    """Fold the batch normalization parameters into the weights for 
       the previous layer."""
    conv_weights = conv_layer.get_weights()

    ## pointwise conv layer has 5 dim. Ex) (1,1,1,3,48) = (batches, conv_x, conv_y, input channels, output channels)
    ## batch norm layer has 2 dim. Ex) (4, 48) = (batch params, output channels)
    ## output layer has 5 dim. Ex) (1,1,1,3,48) = (batches, conv_x, conv_y, input channels, output channels)
    
    # Keras stores the learnable weights for a BatchNormalization layer
    # as four separate arrays:
    #   0 = gamma (if scale == True)
    #   1 = beta (if center == True)
    #   2 = moving mean
    #   3 = moving variancei
    bn_weights = bn_layer.get_weights()

    gamma = bn_weights[0]
    beta = bn_weights[1]
    mean = bn_weights[2]
    variance = bn_weights[3]

    # Now, batch params have # of output channels. Ex) 48

    epsilon = 1e-5
    
    #new_weights = np.array(conv_weights) * gamma / np.sqrt(variance + epsilon)
    new_weights = np.array(conv_weights)

    print("What's the data type?")
    print(new_weights.dtype)

    for i in range(len(bn_weights[0])):
        new_weights[:,:,:,:,i] = np.array(conv_weights)[:,:,:,:,i] * gamma[i] / np.sqrt(variance[i] + epsilon)

    new_bias = beta - mean * gamma / np.sqrt(variance + epsilon)

    # Check upper bound
    up_bound = np.max( [abs(np.max(new_weights)), abs(np.min(new_weights))] )
    print("Pointwise up_bound = ", up_bound)

    # See maximum and minimum weights
    #print("(float32) Maximum new weights = ", np.max(new_weights))
    #print("(float32) Minimum new weights = ", np.min(new_weights))

    # Change float32 to float16 to float32 to drop precisions
    #new_weights = np.float16(new_weights)
    #new_weights = np.float32(new_weights)

    # See maximum and minimum weights
    #print("(float16) Maximum new weights = ", np.max(new_weights))
    #print("(float16) Minimum new weights = ", np.min(new_weights))


    return new_weights, new_bias





def fold_batch_norm(conv_layer, bn_layer):
    """Fold the batch normalization parameters into the weights for 
       the previous layer."""
    conv_weights = conv_layer.get_weights()
    #print("conv_weight shape = ", np.array(conv_weights).shape)

    # Keras stores the learnable weights for a BatchNormalization layer
    # as four separate arrays:
    #   0 = gamma (if scale == True)
    #   1 = beta (if center == True)
    #   2 = moving mean
    #   3 = moving variance
    bn_weights = bn_layer.get_weights()
    #print("bn_weight shape = ", np.array(bn_weights).shape)
    #print("bn_weight gamma = ", np.array(bn_weights[0]))

    gamma = bn_weights[0]
    beta = bn_weights[1]
    mean = bn_weights[2]
    variance = bn_weights[3]
   

    epsilon = 1e-5
    new_weights = conv_weights * gamma / np.sqrt(variance + epsilon)
    #new_weights = conv_weights * gamma / np.sqrt(variance*variance + epsilon)

    new_bias = beta - mean * gamma / np.sqrt(variance + epsilon)
    #new_bias = beta - mean * gamma / np.sqrt(variance*variance + epsilon)

    return new_weights, new_bias

W_nobn = []
W_nobn.extend(fold_batch_norm_depthwise(model.layers[0], model.layers[1]))
W_nobn.extend(fold_batch_norm_pointwise(model.layers[3], model.layers[4]))
W_nobn.extend(fold_batch_norm_depthwise(model.layers[7], model.layers[8]))
W_nobn.extend(fold_batch_norm_pointwise(model.layers[10], model.layers[11]))
W_nobn.extend(fold_batch_norm_depthwise(model.layers[14], model.layers[15]))
W_nobn.extend(fold_batch_norm_pointwise(model.layers[17], model.layers[18]))
W_nobn.extend(fold_batch_norm_depthwise(model.layers[21], model.layers[22]))
W_nobn.extend(fold_batch_norm_pointwise(model.layers[24], model.layers[25]))
W_nobn.extend(fold_batch_norm_depthwise(model.layers[27], model.layers[28]))
W_nobn.extend(fold_batch_norm_pointwise(model.layers[30], model.layers[31]))
W_nobn.extend(model.layers[33].get_weights())

print("Really?")
print(len(W_nobn))
print("W_nobn[0][0] shape", np.array(W_nobn[0][0]).shape)
print("W_nobn[1] shape", np.array(W_nobn[1]).shape)
print("W_nobn[2] shape", np.array(W_nobn[2]).shape)
print("W_nobn[3] shape", np.array(W_nobn[3]).shape)

layer_to_ignore_nobn = [1,3,4,6,8,9,11,13,14,16,18,20,22]
j=0

for i in range(23):
    if i not in layer_to_ignore_nobn:
        #print("W_nobn shape = ", np.array(W_nobn[j]).shape)
        print("i, j", i, j)
        model_no_bn.layers[i].set_weights( [ W_nobn[j][0], W_nobn[j+1] ] )
        #model_no_bn.layers[i].set_weights( np.array( [ W_nobn[j][0] ] ) )
        #model_no_bn.layers[i+1].set_weights( np.array( [ W_nobn[j+1] ] ) )
        j += 2

model_no_bn.layers[23].set_weights( np.array([ W_nobn[20] ]) )

print("Let's take a look at loaded weights")
print(np.array(model_no_bn.get_weights()[2]))

## Comparing models

#print("Comparing models...")
#
#image_data = np.random.random((1, 320, 160, 3))
#features = model.predict(image_data)
#features_nobn = model_no_bn.predict(image_data)
#
#max_error = 0
#for i in range(features.shape[1]):
#    for j in range(features.shape[2]):
#        for k in range(features.shape[3]):
#            diff = np.abs(features[0, i, j, k] - features_nobn[0, i, j, k])
#            max_error = max(max_error, diff)
#            if diff > 1e-4:
#                pass
#                #print(i, j, k, ":", features[0, i, j, k], features_nobn[0, i, j, k], diff)
#
#print("Largest error:", max_error)



model_no_bn.trainable = False

input_img = load_input('images/2.jpg')
output = model_no_bn.predict(input_img).transpose(0,3,1,2)
print(get_box(output))

'''
You should report the average IoU for each quantized model you get
'''
#avg_iou = 0
#for line in lines:
#    input_img = load_input(line[0])
#    output = model_no_bn.predict(input_img).transpose(0,3,1,2)
#    avg_iou+= bbox_iou(get_box(output),line[1])
#avg_iou = avg_iou/1000
#print (avg_iou)
