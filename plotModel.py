# @Date:   2019-12-26T10:27:56+01:00
# @Last modified time: 2020-02-20T20:15:32+01:00

# import sys


from keras.utils import plot_model
import model

import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3#0.41
session = tf.Session(config=config)

numC=17
patch_shape=(32,32,10)

model = model.sen2LCZ_drop(patch_shape, num_classes=17, bn=1, depth=17, dropRate=0.2, fusion=1)
model.summary()
plot_model(model, to_file='./modelFig/'+'sen2LCZ_drop.png', show_shapes='True')
