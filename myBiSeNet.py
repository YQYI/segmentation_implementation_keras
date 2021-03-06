from keras.layers import *
from keras.activations import *
from keras.optimizers import *
from myMobileNetV2 import *

from keras.models import Model
import keras.backend as K

def conv_block(input_data, kernal_n, kernal_s = 3, stride = 1):
    x = Conv2D(
        kernal_n,
        (kernal_s, kernal_s),
        padding = "same",
        strides = stride,
        activation = "relu")(input_data)

    print("conv_block before bn", x.shape)
    x = BatchNormalization(axis = 1)(x)
    return x

def upsampling(input_data,scale):
    x = UpSampling2D(size=(scale, scale), data_format = "channels_last")(input_data)
    return x

class GAP(Layer):
    def __init__(self, data_format=None, **kwargs):
        super(GAP, self).__init__(**kwargs)
        self.data_format = 'channels_first'

    def call(self, inputs):
        return K.mean(inputs, axis=[1, 2], keepdims=True)

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(_GlobalPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])


def ARM(input_data):

    x = GAP()(input_data)
    print("after ARM GAP", x.shape)
    x = Conv2D(x.shape.as_list()[-1], (1, 1), padding = "same", activation = None)(x)
    print("after ARM CON", x.shape)
    #x = BatchNormalization(axis = 1)(x)

    x = sigmoid(x)

    x = multiply([input_data, x])

    return x

def FFM(input_data1, input_data2, num_class):
    x = concatenate([input_data1, input_data2], axis = -1)
    x1 = conv_block(x, num_class, kernal_s = 3)
    x2 = GAP()(x1)
    x2 = Conv2D(num_class, (1, 1), activation = "relu", padding = "same")(x2)
    x2 = Conv2D(num_class, (1, 1), activation = "sigmoid", padding = "same")(x2)
    x3 = multiply([x2, x1])
    x = add([x3, x1])
    return x

def create_BiSeNet(num_class):
    input_data = Input(shape = (321, 321, 3))
    # spatial_path
    spatial_path = conv_block(input_data, kernal_n = 64, kernal_s = 3, stride = 2)
    spatial_path = conv_block(spatial_path, kernal_n = 128, kernal_s = 3, stride = 2)
    spatial_path = conv_block(spatial_path, kernal_n = 256, kernal_s = 3, stride = 2)
    # context_path
    down1, down2 = MobileNetV2(input_data)
    print("original mobilenet result shape:", down1.shape, down2.shape)
    ARM1 = ARM(down1)
    ARM2 = ARM(down2)

    global_ARM2 = GAP()(ARM2)

    ARM2 = multiply([ARM2, global_ARM2])

    #ARM1 = upsampling(ARM1, 2)
    #ARM2 = upsampling(ARM2, 4)

    #context_path = concatenate([ARM1, ARM2], axis = -1)
    #net = FFM(spatial_path, context_path, num_class)
    #net = upsampling(net, 8)
    #net = Conv2D(num_class, (1, 1), activation = None, padding = "same")(net)
    model = Model(inputs = input_data, outputs = down1)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model
