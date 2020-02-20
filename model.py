# @Date:   2019-12-23T11:32:33+01:00
# @Last modified time: 2020-02-20T19:58:25+01:00
# @Date:   2019-12-23T11:24:52+01:00
# @Last modified time: 2020-02-20T19:58:25+01:00



"""
sen2LCZ

"""

from __future__ import print_function
import keras
from keras.layers import *
from keras.regularizers import l2
from keras.models import Model


def sen2LCZ_drop_core(inputs, num_classes=17, bn=1, depth=5, dim=16, dropRate=0.1, fusion=0):

    # Start model definition.
    inc_rate = 2

    lay_per_block=int((depth-1)/4)

    '32*32'
    conv0 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(inputs)

    if bn==1:
        print('with BN')
        conv0 = BatchNormalization(axis=-1)(conv0)
    conv0 = Activation('relu')(conv0)

    for i in np.arange(lay_per_block-1):
        print(str(i) +'in' +str(lay_per_block-1), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        conv0 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv0)
        if bn==1:
            print('with BN')
            conv0 = BatchNormalization(axis=-1)(conv0)
        conv0 = Activation('relu')(conv0)

    "how to pooling?!"
    #############################################
    # merge0 = MaxPooling2D((2, 2))(conv0)

    "original idea"
    pool0 = MaxPooling2D((2, 2))(conv0)
    pool1 = AveragePooling2D(pool_size=2)(conv0)
    merge0 = Concatenate()([pool0,pool1])
    #############################################

    if fusion==1:
        'prediction'
        x = GlobalAveragePooling2D()(merge0)#Flatten
        print(x.shape)
        outputs_32 = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(x)

    '16*16'
    dim=dim*inc_rate
    conv1 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(merge0)
    if bn==1:
        print('with BN')
        conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)

    for i in np.arange(lay_per_block-1):
        print(str(i) +'in' +str(lay_per_block-1), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        conv1 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv1)
        if bn==1:
            print('with BN')
            conv1 = BatchNormalization(axis=-1)(conv1)
        conv1 = Activation('relu')(conv1)

    "how to pooling?!"
    #############################################
    # merge1 = MaxPooling2D((2, 2))(conv1)

    "original idea"
    pool0 = MaxPooling2D((2, 2))(conv1)
    pool1 = AveragePooling2D(pool_size=2)(conv1)
    merge1 = Concatenate()([pool0,pool1])
    #############################################
    'dropOut'
    merge1 = Dropout(dropRate)(merge1)



    if fusion==1:
        'prediction'
        x = GlobalAveragePooling2D()(merge1)#Flatten
        print(x.shape)
        outputs_16 = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(x)

    '8*8'
    dim=dim*inc_rate
    conv2 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(merge1)
    if bn==1:
        print('with BN')
        conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)

    for i in np.arange(lay_per_block-1):
        print(str(i) +'in' +str(lay_per_block-1), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        conv2 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv2)
        if bn==1:
            print('with BN')
            conv2 = BatchNormalization(axis=-1)(conv2)
        conv2 = Activation('relu')(conv2)


    "how to pooling?!"
    #############################################
    # merge2 = MaxPooling2D((2, 2))(conv2)

    "original idea"
    pool0 = MaxPooling2D((2, 2))(conv2)
    pool1 = AveragePooling2D(pool_size=2)(conv2)
    merge2 = Concatenate()([pool0,pool1])
    #############################################

    'dropOut'
    merge2 = Dropout(dropRate)(merge2)

    if fusion==1:
        'prediction'
        x = GlobalAveragePooling2D()(merge2)#Flatten
        print(x.shape)
        outputs_8 = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(x)

    '4*4'
    dim=dim*inc_rate
    conv3 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(merge2)
    if bn==1:
        print('with BN')
        conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)

    for i in np.arange(lay_per_block-1):
        print(str(i) +'in' +str(lay_per_block-1), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        conv3 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv3)
        if bn==1:
            print('with BN')
            conv3 = BatchNormalization(axis=-1)(conv3)
        conv3 = Activation('relu')(conv3)

    'prediction'
    x = GlobalAveragePooling2D()(conv3)#Flatten
    print(x.shape)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)

    if fusion==1:
        'prediction'
        x = GlobalAveragePooling2D()(merge2)#Flatten
        print(x.shape)
        outputs_8 = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(x)
        o=outputs=Average()([outputs, outputs_32, outputs_16, outputs_8])
    else:
        o=outputs

    return o

def sen2LCZ_drop(input_shape, num_classes=17, bn=1, depth=5, dim=16, dropRate=0.1, fusion=0):
    """

    # Arguments

    # Returns
        model (Model): Keras model instance
    """

    inputs = Input(shape=input_shape)
    o=sen2LCZ_drop_core(inputs, num_classes=num_classes, bn=bn, depth=depth, dim=dim, dropRate=dropRate, fusion=fusion)

    return Model(inputs=inputs, outputs=o)
