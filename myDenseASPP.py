from keras.layers import *
from keras.activations import *
from myMobileNetV2 import *

from keras.models import Model
import keras.backend as K

def dilated_block(input_data, kernal_num, rate = 1, kernal_size = [3, 3], stride = 1):
    x = BatchNormalization(axis = 1)(input_data)
    x = Conv2D(kernal_num, kernal_size,
               padding = "same", strides = stride,
               activation = "relu", dilation_rate = rate)(input_data)
    return x

def upsampling(input_data,scale):
    input_shape = input_data.shape.as_list()
    return K.resize_images(input_data, scale, scale, "channels_last")

def create_DenseASSPNet(num_class):
    input_data = Input(shape = (512, 512, 3))
    print(type(input_data))
    # spatial_path
    spatial_path = conv_block(input_data, kernal_n = 64, kernal_s = 3, stride = 2)
    print(type(spatial_path))
    spatial_path = conv_block(spatial_path, kernal_n = 128, kernal_s = 3, stride = 2)
    spatial_path = conv_block(spatial_path, kernal_n = 256, kernal_s = 3, stride = 2)
    # context_path
    down1, down2 = MobileNetV2(input_data)
    print(down1.shape, down2.shape)
    ARM1 = ARM(down1)
    ARM2 = ARM(down2)

    global_ARM2 = GAP()(ARM2)

    ARM2 = multiply([ARM2, global_ARM2])

    ARM1 = Lambda(upsampling, arguments = {'scale' : 2})(ARM1)
    ARM2 = Lambda(upsampling, arguments = {'scale' : 4})(ARM2)
    context_path = concatenate([ARM1, ARM2], axis = -1)
    net = FFM(spatial_path, context_path, num_class)
    net = Lambda(upsampling, arguments = {'scale' : 8})(net)
    net = Conv2D(num_class, (1, 1), activation = None, padding = "same")(net)
    model = Model(inputs = input_data, outputs = net)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
