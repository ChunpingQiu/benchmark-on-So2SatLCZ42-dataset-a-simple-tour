# @Date:   2018-07-16T20:51:18+02:00
# @Email:  chunping.qiu@tum.de
# @Last modified time: 2020-01-10T10:33:19+01:00


import sys
import os
import numpy as np
from skimage.util.shape import view_as_windows
import glob
from osgeo import gdal,osr
import glob2
from scipy import stats
import scipy.ndimage
from memprof import *
from sklearn.preprocessing import StandardScaler

'load img tif and get patches from it;  save as tif file after getting predictions'
class img2mapC(object):

  def __init__(self, dim_x, dim_y, dim_z, step, Bands, scale, nanValu, dim_x_img, dim_y_img):
	  self.dim_x = dim_x#shape of the patch to input the model
	  self.dim_y = dim_y

	  self.step = step#lcz resolution (in pixel): step when cropping patches from the image

	  self.Bands = Bands#bands selected from the image files, list

	  self.scale = scale#the number used to divided the pixel value by

	  self.nanValu = nanValu

	  self.dim_x_img = dim_x_img#the actually extracted image patch, can be different from the patch size input to the network
	  self.dim_y_img = dim_y_img

  '''
	# cut a matrix into patches
	# input:
			imgMat: matrix containing the bands of the image
			upSampleR: ratio to sample the mat
	# output:
			patch: the patches, one patch is with the shape: dim_x, dim_y, dim_z
			R: the size of the final lcz map
			C: the size of the final lcz map
			idxNan: the index of no data area.
  '''
  def Bands2patches(self, imgMat, upSampleR):

	  print('imgMat', imgMat.shape, imgMat.dtype)

	  "for each band"
	  for band in np.arange(imgMat.shape[2]):

		  arr = imgMat[:,:,band]
		  if upSampleR>1.1:
					arr=scipy.ndimage.zoom(arr, [upSampleR,  upSampleR], order=1)#Bilinear interpolation would be order=1

		  patch0, R, C= self.__img2patch(arr)#'from band to patches'

		  if band==0:

			  "find the nodata area (0 pixel value)"
			  patch0Tmp=np.amin(patch0, axis=1);
			  indica=np.amin(patch0Tmp, axis=1);

			  idxNan = np.where( (indica<0.000001) )
			  idxNan = idxNan[0].reshape((-1,1))
			  print("no data pixels: ", idxNan.shape)

			  patch=np.zeros(((patch0.shape[0]-idxNan.shape[0]), self.dim_x, self.dim_y, imgMat.shape[2]), dtype=imgMat.dtype);

		  patch0 = np.delete(patch0, idxNan, axis=0)

		  "scale with a fucntion"
		  if self.scale == -1:
			  patch[:,:,:,band]=self.scaleBand(patch0)
		  else:#"simple scale"
			  patch[:,:,:,band]=patch0/self.scale ;

	  return patch, R, C, idxNan


# # #from a multi bands mat to patches, without considering the nan area
#   def Bands2patches_all(self, imgMat, upSampleR):
# 	  for band in np.arange(imgMat.shape[2]):
#
# 		  arr = imgMat[:,:,band]
# 		  if upSampleR!=1:
# 					arr=scipy.ndimage.zoom(arr, [upSampleR,  upSampleR], order=1)#Bilinear interpolation would be order=1
# 		  patch0, R, C= self.__img2patch(arr)#'from band to patches'
#
# 		  if band==0:
# 			  patch=np.zeros(((patch0.shape[0]), self.dim_x, self.dim_y, imgMat.shape[2]), dtype=imgMat.dtype);
#
# 		  #print('self.scale', self.scale)
# 		  if self.scale == -1:#scale with a fucntion
# 			  patch[:,:,:,band]=self.scaleBand(patch0)
#
# 		  else:
# 			  patch[:,:,:,band]=patch0/self.scale ;
#
# 	  return patch, R, C


  '''
	# load all relevent bands of a image file
	# input:
			imgFile: image file
	# output:
			prj: projection data
			trans: projection data
			matImg: matrix containing the bands of the image
  '''
  def loadImgMat(self, imgFile):

	  src_ds = gdal.Open( imgFile )

	  if src_ds is None:
		  print('Unable to open INPUT.tif')
		  sys.exit(1)

	  prj=src_ds.GetProjection()
	  trans=src_ds.GetGeoTransform()

	  bandInd=0
	  print(self.Bands)
	  for band in self.Bands:
		  band += 1
		  srcband = src_ds.GetRasterBand(band)

		  if srcband is None:
			  print('srcband is None'+str(band)+imgFile)
			  continue

		  #print('srcband read:'+str(band))

		  arr = srcband.ReadAsArray()
		  #print(np.unique(arr))

		  if bandInd==0:
			  R=arr.shape[0]
			  C=arr.shape[1]
			  #print(arr.shape)
			  matImg=np.zeros((R, C, len(self.Bands)), dtype=np.float32);

		  matImg[:,:,bandInd]=np.float32(arr)

		  bandInd += 1

	  return prj, trans, matImg


  # '''
	#  # create a list of files dir for all the cities needed to be produced
	#  # input:
	# 		 fileD: cities path
	# 		 cities: a list of cities under the fileD
	#  # output:
	# 		 files: all the files
	# 		 imgNum_city: the image number of each city
  # '''
  # def createFileList_cities(self, fileD, cities):
	#   files = []
	#   imgNum_city = np.zeros((len(cities),1), dtype=np.uint8)
	#   for j in np.arange(len(cities)):
	# 		 #all seasons
	# 		 file = sorted(glob2.glob(fileD+ cities[j] +'/**/*_'  + '*.tif'))
	# 		 files.extend(file)
	# 		 imgNum_city[j] = len(file)
  #
	#   return files, imgNum_city

  '''
	  # create a list of files dir for all the images in different seasons of the input city dir
	  # input:
			  fileD: the absolute path of one city
	  # output:
			  files: all the files corresponding to different seasons
  '''
  def createFileList(self, fileD):
	  files = []
	  imgNum_city = np.zeros((1,1), dtype=np.uint8)

	 #all seasons
	  file = sorted(glob2.glob(fileD +'/**/*_'  + '*.tif'))
	  files.extend(file)
	  return files

  '''
  'use the imgmatrix to get patches'
	input:
		   mat: a band of the image
	output:
			patches: the patch of this input image, to feed to the classifier
			R: the size of the final lcz map
			C: the size of the final lcz map
  '''
  def  __img2patch(self, mat):

	  window_shape = (self.dim_x_img, self.dim_y_img)

	  B = view_as_windows(mat, window_shape, self.step)
	  #print(B.shape)

	  patches=np.reshape(B, (-1, window_shape[0], window_shape[1]))
	  #print(patches.shape)

	  #the size of the final map
	  "one patch one output prediciton"
	  R=B.shape[0]
	  C=B.shape[1]

	  return patches, R, C


  '''
	# save prediction as tif
	# input:
			yPre0: the vector of the predictions
			R: the size of the final lcz map
			C: the size of the final lcz map
			prj: projection data
			trans: projection data
			mapFile: the file to save the produced map
			idxNan: the index of no data area.
	# output:
			no
  '''
  def predic2tif_vector(self, yPre0, R, C, prj, trans, mapFile, idxNan):

	  totalNum=R*C;

	  "setting the projection of the output file"
	  xres =trans[1]*self.step
	  yres= trans[5]*self.step

	  "assuming that trans[0] is the center of the first pixel"
	  geotransform = (trans[0] + trans[1]*(self.dim_x_img-1)/2.0, xres, 0, trans[3] + trans[5]*(self.dim_y_img-1)/2.0, 0, yres)

	  dimZ=np.shape(yPre0)[1]

	 # create the dimZ raster file
	  dst_ds = gdal.GetDriverByName('GTiff').Create(mapFile, C, R, dimZ, gdal.GDT_UInt16)#gdal.GDT_Byte .GDT_Float32 gdal.GDT_Float32
	  dst_ds.SetGeoTransform(geotransform) # specify coords
	  dst_ds.SetProjection(prj)

	  for i in np.arange(dimZ):

		  yPre = np.zeros((totalNum, 1), dtype=np.uint16 ) + self.nanValu + 1;
		  yPre[idxNan]= self.nanValu;# set no data value
		  tmp = np.where( (yPre== self.nanValu + 1 ) )

		  yPre[ tmp[0]]=yPre0[:,i].reshape((-1,1));

		  map=np.reshape(yPre, (R, C))
		  dst_ds.GetRasterBand(int(i+1)).WriteArray(map)   # write band to the raster

	  dst_ds.FlushCache()                     # write to disk
	  dst_ds = None

  # '''
	# # save a map as tif
	# # input:
	# 		mat: the matrix to be saved
	# 		prj: projection data
	# 		trans: projection data
	# 		mapFile: the file to save the produced map
	# # output:
	# 		no
  # '''
  # def predic2tif(self, mat, prj, trans, mapFile):
  #
	#   R=mat.shape[0]
	#   C=mat.shape[1]
  #
  #     "setting the projection of the output file"
	#   xres =trans[1]*self.step
	#   yres= trans[5]*self.step
  #
  #     "assuming that trans[0] is the center of the first pixel"
	#   geotransform = (trans[0] + trans[1]*(self.dim_x_img-1)/2.0, xres, 0, trans[3] + trans[5]*(self.dim_y_img-1)/2.0, 0, yres)
  #
	#   dimZ=mat.shape[2]
  #
	#  # create the dimZ raster file
	#   dst_ds = gdal.GetDriverByName('GTiff').Create(mapFile, C, R, dimZ, gdal.GDT_UInt16)#gdal.GDT_Byte .GDT_Float32 gdal.GDT_Float32
	#   dst_ds.SetGeoTransform(geotransform) # specify coords
	#   dst_ds.SetProjection(prj)
  #
	#   for i in np.arange(dimZ):
	# 	  map=mat[:,:,i]
	# 	  dst_ds.GetRasterBand(int(i+1)).WriteArray(map)   # write band to the raster
  #
	#   dst_ds.FlushCache()                     # write to disk
	#   dst_ds = None

  '''
	# from a image to one proba file, returned
	# input:
			file: the file of the image
			model: the trained model
	# output:
			y_pre0: the prob from the model prediction.
			mapR, mapC: the size of the prediciton (not the image, as one patch outpus one label)
			prj, trans
			idxNan: no output for this patch as there are no data areas within this patch

  '''
  def file2prediction(self, file, model):
	  prj, trans, img= self.loadImgMat(file)

	  # R=img.shape[0]
	  # C=img.shape[1]

	  upSampleR = self.dim_x/self.dim_x_img
	  print("upSampleR = self.dim_x/self.dim_x_img: ", upSampleR)

	  x_test, mapR, mapC, idxNan = self.Bands2patches(img, upSampleR)

	  print('x_test shape:', x_test.shape)

	  y_pre0 = model.predict(x_test, batch_size = 16, verbose=1)

	  return y_pre0, mapR, mapC, prj, trans, idxNan

  # def file2prediction_(self, file, model):
	# 	  prj, trans, img= self.loadImgMat(file)
	# 	  R=img.shape[0]
	# 	  C=img.shape[1]
	# 	  print('img:', R, C)
	# 	  x_test, mapR, mapC = self.Bands2patches_all(img,1)
	# 	  print('x_test:', x_test.shape)
	# 	  y_pre0 = model.predict(x_test, batch_size = 128, verbose=1)
  #
	# 	  return y_pre0, mapR, mapC, prj, trans

  '''
	# from seasons to one map and several proba files, returned
	# input:
			files: the files of the images in this season
			model: the trained model
			proFile: the path (and name) to save the prob.
	# output:
			no
  '''
  def season2map_(self, files, model, proFile):

	  numImg=len(files);
	  for idSeason in np.arange(numImg):
		  file = files[idSeason]
		  print("processing image: ", file)

		  y_pre0, mapR, mapC, prj, trans, idxNan = self.file2prediction(file, model)
		  print('map size:', mapR, mapC)

		  y_pre = self.predict_classes((y_pre0))
		  print("existing classes: ", np.unique((np.int8(y_pre))))

		  y_pre=y_pre.reshape(-1,1) ;

		  '''save soft prob'''
		  proFile0 = proFile+file[file.rfind('_'):]
		  self.predic2tif_vector(y_pre0*10000, mapR, mapC, prj, trans, proFile0, idxNan)

		  totalNum=mapR*mapC;
		  if idSeason==0:
			  yPreAll=np.empty((totalNum, np.int8(numImg)))

		  yPre=np.zeros((totalNum,1), dtype=np.int8 ) + self.nanValu + 1;
		  yPre[idxNan]= self.nanValu;
		  tmp = np.where( (yPre== self.nanValu + 1 ) )

		  "setting nan value"
		  # yPre=np.zeros((totalNum,1), dtype=np.int8 ) ;
		  # yPre[idxNan]= nan ;
		  # tmp = np.where( (yPre<= 0.1 ) )
		  yPre[ tmp[0]]=y_pre.reshape((-1,1)) ;

		  yPreAll[:,idSeason]=yPre.reshape(yPre.shape[0]);

	  '''majority voting of seasons'''
	  m = stats.mode(yPreAll, 1)
	  print("type of m: ", yPreAll.dtype)
	  print("existing classes: ", np.unique((m[0])))

	  return prj, trans, np.reshape(m[0], (mapR, mapC))

  '''
	# from a season to one map and several proba files, saved
	# input:
			files: the files of the images in this season
			model: the trained model
			proFile: the path (and name) to save the prob.
			mapFile: the file to save the produced map
	# output:
			no
  '''
  #@memprof(plot = True)
  def season2map(self, files, model, proFile, mapFile):

	  prj, trans, map=self.season2map_(files, model, proFile)

	  # self.predic2tif(mapConfi, prj, trans, mapFile + '.tif')

	  R = map.shape[0]
	  C = map.shape[1]

	  "setting the projection of the output file"
	  xres =trans[1]*self.step
	  yres= trans[5]*self.step

	  "assuming that trans[0] is the center of the first pixel"
	  geotransform = (trans[0] + trans[1]*(self.dim_x_img-1)/2.0, xres, 0, trans[3] + trans[5]*(self.dim_y_img-1)/2.0, 0, yres)

	 # create the dimZ raster file
	  LCZFile = gdal.GetDriverByName('GTiff').Create(mapFile + '.tif', C, R, 1, gdal.GDT_Byte)#gdal.GDT_Byte .GDT_Float32 gdal.GDT_Float32
	  LCZFile.SetGeoTransform(geotransform) # specify coords
	  LCZFile.SetProjection(prj)

	  # save file with predicted label
	  outBand = LCZFile.GetRasterBand(1)

	  # create color table
	  colors = gdal.ColorTable()

	  # set color for each value
	  colors.SetColorEntry(1, (165,   0,       33))
	  colors.SetColorEntry(2, (204,   0,        0))
	  colors.SetColorEntry(3, (255,   0,        0))
	  colors.SetColorEntry(4, (153,   51,       0))
	  colors.SetColorEntry(5, (204,   102,      0))

	  colors.SetColorEntry(6, (255,   153,      0))
	  colors.SetColorEntry(7, (255,   255,      0))
	  colors.SetColorEntry(8, (192,   192,    192))
	  colors.SetColorEntry(9, (255,   204,    153))
	  colors.SetColorEntry(10, (77,    77,     77))

	  colors.SetColorEntry(11, (0,   102,      0))
	  colors.SetColorEntry(12, (21,   255,     21))
	  colors.SetColorEntry(13, (102,   153,      0))
	  colors.SetColorEntry(14, (204,   255,    102))
	  colors.SetColorEntry(15, (0,     0,    102))

	  colors.SetColorEntry(16, (255,   255,    204))
	  colors.SetColorEntry(17, (51,   102,    255))

	  # set color table and color interpretation
	  outBand.SetRasterColorTable(colors)
	  outBand.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

	  outBand.WriteArray(map)
	  outBand.FlushCache()
	  del(outBand)



  ''' generate class prediction from the input samples'''
  def predict_classes(self, x):
	  y=x.argmax(axis=1)+1
	  return y





#
#   def season2Bdetection(self, files, model, proFile, mapFile):
# 	  prj, trans, mapConfi=self.season2map_(files, model, proFile)
# 	  mapConfi[mapConfi== 2] = self.nanValu
# 	  self.predic2tif(mapConfi, prj, trans, mapFile+'.tif')
#
#   def scaleBand(self,patches):
#       patches_=np.zeros(patches.shape, dtype=np.float32)
#       #for b in np.arange(patches.shape[-1]):
#
#       patch=patches.reshape(-1,1)
#         #print(patch.shape)
#       scaler = StandardScaler().fit(patch)
#         #print(scaler.mean_.shape)
#       patches_=scaler.transform(patch).reshape(patches.shape[0],patches.shape[1], patches.shape[2])
#
#       return patches_
#
# #
#   def img2Bdetection(self, file, model, proFile, mapFile):
#       y_pre0, mapR, mapC, prj, trans = self.file2prediction_(file[0], model)
#
#       print('y_pre0.shape', y_pre0.shape)
#       y=y_pre0.argmax(axis=3)+1
#       print(np.unique(y))
#       y[y== 2] = self.nanValu#to make sure only the buildings are labeled
#       print('y.shape', y.shape)
#       del y_pre0
#
#
#       # prj, trans, img= self.loadImgMat(file[0])
#       # R=img.shape[0]
#       # C=img.shape[1]
#       # print('img:', R, C)
#       # x_test, mapR, mapC = self.Bands2patches_all(img,1)
#       # y=x_test[:,:,:,0]*10000
#
#       mapPatch_shape=y.shape[1]
#       B_=np.reshape(y, (mapR, mapC, y.shape[1], y.shape[2]))
#       print('B_.shape', B_.shape)
#       del y
#
#       C=B_.transpose(0,2,1,3).reshape(-1,B_.shape[1]*B_.shape[3])
#       print('C.shape', C.shape)
#       del B_
#
#       mapConfi=np.zeros((C.shape[0], C.shape[1], 1), dtype=np.int8)
#       mapConfi[:,:,0]=C;
#
#       if mapPatch_shape*2==self.dim_x:
#           print('downsampling by 2!')
#           trans0 =trans[0]+trans[1]*(2-1)/2.0
#           trans3= trans[3]+trans[5]*(2-1)/2.0
#           trans1 =trans[1]*2
#           trans5= trans[5]*2
#           trans = (trans0, trans1, 0, trans3, 0, trans5)
#       self.predic2tif(mapConfi, prj, trans, mapFile+'.tif')
#
#   def img2Bdetection_ovlp(self, file, model, mapFile):
#       prj, trans, img= self.loadImgMat(file[0])
#       R=img.shape[0]
#       C=img.shape[1]
#       print('img:', R, C)
#
#       imgN=len(file);
#       # imgAll=np.zeros((imgN, R, C,img.shape[2]), , dtype=np.float32)
#       # for i in np.arange(imgN):
#       #     prj, trans, img0= self.loadImgMat(file[i])
#       #     imgAll[i,:,:,:]=img0
#
#       if self.dim_x_img==32:
#           paddList=[0,8,16,24]
#       if self.dim_x_img==64:
#           paddList=[0,16,32,48]
#       else:
#           paddList=[0,32,64,96]
#
#
#       for padding in paddList:#[0,16,32,48,64,80,96,112]
#
#           if imgN==1:
#               if padding==0:
#                   img1=img
#               else:
#                   img1=np.pad(img, ((padding, 0), (padding, 0), (0,0)), 'reflect')
#               print(img1.shape)
#               x_test, mapR, mapC = self.Bands2patches_all(img1,1)
#               print('x_test:', x_test.shape)
#               y = model.predict(x_test, batch_size = 4, verbose=1)
#           if imgN==4:
#               prj, trans, img_1= self.loadImgMat(file[1])
#               prj, trans, img_2= self.loadImgMat(file[2])
#               prj, trans, img_3= self.loadImgMat(file[3])
#               if padding==0:
#                   img0=img
#                   img1=img_1
#                   img2=img_2
#                   img3=img_3
#               else:
#                   img0=np.pad(img, ((padding, 0), (padding, 0), (0,0)), 'reflect')
#                   img1=np.pad(img_1, ((padding, 0), (padding, 0), (0,0)), 'reflect')
#                   img2=np.pad(img_2, ((padding, 0), (padding, 0), (0,0)), 'reflect')
#                   img3=np.pad(img_3, ((padding, 0), (padding, 0), (0,0)), 'reflect')
#               #print(img1.shape)
#               x_test0, mapR, mapC = self.Bands2patches_all(img0,1)
#               x_test1, mapR, mapC = self.Bands2patches_all(img1,1)
#               x_test2, mapR, mapC = self.Bands2patches_all(img2,1)
#               x_test3, mapR, mapC = self.Bands2patches_all(img3,1)
#               print('x_test0:', x_test0.shape)
#               y = model.predict([x_test0, x_test1, x_test2, x_test3], batch_size = 4, verbose=1)
#
#
#           mapPatch_shape=y.shape[1]
#           print('class num:', y.shape[-1])
#
#           B_=np.reshape(y, (mapR, mapC, y.shape[1], y.shape[2], y.shape[-1]))
#           print('B_.shape', B_.shape)
#           del y
#
#           C=np.zeros((B_.shape[0]*B_.shape[2], B_.shape[1]*B_.shape[3], B_.shape[4]), dtype=float)
#           for dim in np.arange(B_.shape[4]):
#               B_1=B_[:,:,:,:,dim]
#               C[:,:,dim]=B_1.transpose(0,2,1,3).reshape(-1,B_1.shape[1]*B_1.shape[3])
#               #
#               del B_1
#
#           print('the prediction size', C.shape)
#           OS = np.int( self.dim_x_img/ mapPatch_shape )   #ratio between the input and the output
#
#           if padding==0:
#               r=C.shape[0]
#               c=C.shape[1]
#               Pro=C[0:(r-mapPatch_shape),0:(c-mapPatch_shape),:]
#               print('the final size of the prediction:', Pro.shape)
#               print('uniue(y):', np.unique(Pro.argmax(axis=2)))#np.unique
#           else:
#               Pro=Pro+C[np.int(padding/OS):(r-mapPatch_shape+np.int(padding/OS)), np.int(padding/OS):(c-mapPatch_shape+np.int(padding/OS)), :]
#
#
#     #print('y_pre0.shape', y_pre0.shape)
#       y=Pro.argmax(axis=2)+1
#       #y[y== 2] = 0#self.nanValu#to make sure only the buildings are labeled
#
#       mapConfi=np.zeros((y.shape[0], y.shape[1], 1), dtype=np.int16)
#       mapConfi[:,:,0]=y;
#
#       mapPro=np.zeros((y.shape[0], y.shape[1], 1), dtype=np.int16)
#       mapPro= Pro*10000;
#
#       if mapPatch_shape*2==self.dim_x:
#           print('downsampling by 2!')
#           trans0 =trans[0]+trans[1]*(2-1)/2.0
#           trans3= trans[3]+trans[5]*(2-1)/2.0
#           trans1 =trans[1]*2
#           trans5= trans[5]*2
#           trans = (trans0, trans1, 0, trans3, 0, trans5)
#       self.predic2tif(mapConfi, prj, trans, mapFile+'.tif')
#       self.predic2tif(mapPro, prj, trans, mapFile+'_pro.tif')
#
#       return y, prj, trans
