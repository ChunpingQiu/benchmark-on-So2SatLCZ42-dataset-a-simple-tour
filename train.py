# @Date:   2020-02-04T17:07:00+01:00
# @Last modified time: 2020-02-20T20:15:52+01:00

import resnet
import model
import lr

from dataLoader import generator

from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
# from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45#0.41
session = tf.Session(config=config)

###################################################
'path to save:'
file0='./'

'path to data'
train_file='/work/qiu/SEN2_LCZ42_UTM/TIANCHI_FINAL_DATASET/training.h5'
validation_file='/work/qiu/SEN2_LCZ42_UTM/TIANCHI_FINAL_DATASET/validation.h5'

patch_shape=(32,32,10)
numClasses=17
batchSize=32

trainNumber=352366
validationNumber=24119
lr_sched = lr.step_decay_schedule(initial_lr=0.002, decay_factor=0.5, step_size=5)

###################################################
# model = resnet.resnet_v2(input_shape=patch_shape, depth=11, num_classes=numClasses)
model = model.sen2LCZ_drop(patch_shape, num_classes=numClasses, bn=1, depth=17, dropRate=0.2, fusion=1)

model.compile(optimizer = Nadam(), loss = 'categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 40)
modelbest = file0 + "_" + str(batchSize) +"_weights.best.hdf5"
checkpoint = ModelCheckpoint(modelbest, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

model.fit_generator(generator(train_file, batchSize=batchSize, num=trainNumber),
                steps_per_epoch = trainNumber//batchSize,
                validation_data= generator(validation_file, num=validationNumber, batchSize=batchSize),
                validation_steps = validationNumber//batchSize,
                epochs=100,
                max_queue_size=100,
                callbacks=[early_stopping, checkpoint, lr_sched])
