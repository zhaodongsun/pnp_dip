# Plug-and-play Deep Image Prior
This repository contains the code for plug-and-play deep image prior.

## Requirement
+ pytorch==1.2.0
+ numpy
+ matplotlib
+ skimage
+ glob
+ bm3d
+ prox_tv

## Usage
The directory `data` contains 25 images from CelebA dataset. We use these data to get the PSNR curves and table in the paper. The directory `data_visual` has *pepper* and *monarch*  images, which are used for visual results in the paper.


You can directly run the python files in 
+ `experiments/inpainting`
+ `experiments/inpainting_vis`
+ `experiments/uniform_denoising`
+ `experiments/uniform_denoising_vis`
+ `experiments/superrsolution_vis`

The names of python files are the priors they use. For DIP + others, it use our plug-and-play deep image prior method. Without DIP in the name, it uses plug-and-play prior method.

When the directory has the suffix `vis`, it will use the data at `data_visual`. Otherwise, it will use the data at `data`. The result will be stored in the corresponding data directory.
