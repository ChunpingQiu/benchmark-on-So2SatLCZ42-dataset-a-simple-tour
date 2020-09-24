# @Date:   2018-07-16T20:51:19+02:00
# @Email:  chunping.qiu@tum.de
# @Last modified time: 2018-08-10T12:23:29+02:00



# File              : img2lczMap_oneCity.py
# Author            : Chunping Qiu
# Date              : 07.07.2018 12:07:22
# Last modified: 07.07.2018 12:07:27 Chunping Qiu
# first version, fit to global processing

# import cProfile
import sys
import numpy as np
import time
import os

sys.path.insert(0, '../')
import model

from keras.models import load_model
from img2mapC4Lcz import img2mapC

import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)

numClass= 17
step=10#the step for map production
patch_shape = (32, 32, 10)

params = {'dim_x': patch_shape[0],
		   'dim_y': patch_shape[1],
		   'dim_z': patch_shape[2],
		   'step': step,
		   'Bands': [1,2,3,4,5,6,7,8,11,12],#band index starts from 0, #corresponding to the bands in the training dataset
		   'scale':10000.0,
		   # 'isSeg':0,
		   'nanValu': 0,
		   'dim_x_img': patch_shape[0],#the actuall extracted image patch
		   'dim_y_img': patch_shape[1]}

#model path
modelFile=sys.argv[1]
# modelFile="/home/qiu/CodeSummary/0urbanMapper/so2satLCZ42/results/_32_weights.best.hdf5"

#image path
fileD = sys.argv[2]#os.getcwd()
# fileD = "/work/qiu/LCZ42_GEE/00017_22007_Lagos"
print(fileD)

MapfileD=fileD+'/LCZ_results/'; # save the results under the city folder
MapfileD_pro=MapfileD; # save the results under the city folder

if not os.path.exists(MapfileD):
    os.makedirs(MapfileD)
if not os.path.exists(MapfileD_pro):
    os.makedirs(MapfileD_pro)

img2mapCLass=img2mapC(**params);

#find the image files for this city (may have 4 seasons or less)
files=img2mapCLass.createFileList(fileD) ;
#print(files)

#exclude the files which are not images for production
image2processed=[ x for x in files if "LCZ_results" not in x ];
print(image2processed)

numImg=len(image2processed)
#print(numImg)
if numImg==0:
	sys.exit("no images found!")

city=fileD[fileD.rfind('/')+1:]
#print(city)

#load the model

# model = resnet_v2.resnet_v2(input_shape = patch_shape, depth = 20, num_classes = 17)
model = model.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1)

print(modelFile)
model.load_weights(modelFile, by_name=False)

mapFile=MapfileD+city
mapFile_pro=MapfileD_pro+city

img2mapCLass.season2map(image2processed, model, mapFile_pro, mapFile)
