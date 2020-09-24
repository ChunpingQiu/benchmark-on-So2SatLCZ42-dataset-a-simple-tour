# LCZ classification from so2satLCZ42 dataset

## basics and requirements

keras (tensorflow backend)

data:
https://arxiv.org/abs/1912.12171

http://doi.org/10.14459/2018MP1454690


## Folder Structure
  ```
  pytorch-template/
  ├── train.py - main file for training (path to data needs to be set)
  ├── dataLoader.py - loading data from h5 files
  ├── model.py - architecture of sen2LCZ_drop
  ├── evaluation.py - evaluation of the trained models
  ├── lr.py - learning rate schedule
  ├── plotModel.py - plot the models
  │
  │
  │
  ├── results/ - (temporary) results folder
  │   ├── plotModel.py - plot the models
  │   ├── modelS.py - select the models according to the setup(single task or mtl)
  │   ├── model_sep_cbam.py - definition of the mtl framework
  │   └── ...
  │   
  │    
  ├── img2map/ - predict using the trained models from s2 data
  │   ├── img2lczMap_oneCity.py - read s2 data and predict and save the results in geotiff
  │   ├── img2mapC4Lcz.py - functions for predictions
  │   └── ...
  │   
  │
  └── modelFig/ - figure of the model structure
      ├──  
      ├──
      └── ...       

  ```
## Usage

### img2map
- setting model path and image path, [image data for test](https://drive.google.com/drive/u/1/folders/1y-lFSuUeY3barjKJVG1TTwh39RqlUzn6)
- `CUDA_VISIBLE_DEVICES=0 python img2lczMap_oneCity.py "../results/_32_weights.best.hdf5" "testData/00017_22007_Lagos"`

### train
- after setting the data path (in train.py): `CUDA_VISIBLE_DEVICES=0 python train.py`


<!-- ## td list -->
<!---
[//]: # (- [x] predict with the trained model)
- [x] test different models with the same data
- [x] training different models under the same configuration
- [x] check created patches
- [x] from images to patches
-->
