# @Date:   2020-04-24T10:19:05+02:00
# @Last modified time: 2020-04-25T11:59:59+02:00

import model

import numpy as np
import h5py
import scipy.io as scio

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
session = tf.Session(config=config)

def predata4LCZ(file, keyX, keyY):
    hf = h5py.File(file, 'r')
    x_tra = np.array(hf[keyX])
    y_tra = np.array(hf[keyY])
    hf.close()

    print(x_tra.shape, y_tra.shape)

    return x_tra, y_tra
################################################################################
file0 ='./results/'
model = model.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1)
batch_size = 32#8 16 32
numC= 17 ;

'loading test data'
file='/work/qiu/SEN2_LCZ42_UTM/TIANCHI_FINAL_DATASET/testing.h5'
x_tst, y_tst= predata4LCZ(file, 'sen2', 'label')
patch_shape = (32, 32, 10)


#########################################
modelbest = file0  + "_" + str(batch_size) +"_weights.best.hdf5"


'load saved best model'
model.load_weights(modelbest, by_name=False)

# 4. test phase
y_pre = model.predict(x_tst, batch_size = batch_size)
y_pre = y_pre.argmax(axis=-1)+1
y_testV = y_tst.argmax(axis=-1)+1


C = confusion_matrix(y_testV-1, y_pre-1, labels=np.arange(numC))
print('#####################confusion_matrix:')
print(C)
# print(type(C))

classRep = classification_report(y_testV, y_pre)
oa = accuracy_score(y_testV, y_pre)
cohKappa = cohen_kappa_score(y_testV, y_pre)

print('#####################classwise accuracy:')
print(classRep)

print('#####################overall accuracy and Kappa:')
print(oa, cohKappa)

scio.savemat((file0 + 'Acc' + "_" + str(batch_size)+'.mat'), {'classRep': classRep ,'oa': oa, 'cohKappa': cohKappa, 'confusion_matrix': np.int64(C), 'y_testV':y_testV, 'y_pre':y_pre})
#
