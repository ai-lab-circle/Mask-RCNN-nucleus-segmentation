# Mask R-CNN for nucleus Detection and Segmentation (in progress)

This source includes nucleui segmentation code using Mask-RCNN.   

Code was implemented on ubuntu 16.04, python3.5, CUDA9.0, and tensorflow1.12.0.



## Dataset

Download [preprocessed images](https://drive.google.com/file/d/1uF_hxZJZTh8eWSwYuCP8gMGQw3ZxYEwD/view?usp=sharing) for segmentation   


```bash
   mv downloaded_images  MASK_RCNN_ROOT/datasets/nucleus
```

(optional) Download [original nucleus dataset](http://andrewjanowczyk.com/wp-static/nuclei.tgz)

## Model

Download [pretrained model](https://drive.google.com/drive/folders/1SF2727HImKzzhWZ_cCmJkhf2sYenyxgi?usp=sharing)

```bash
   mv downloaded_model  MASK_RCNN_ROOT/logs/nucleus20190130T0908/
```


## (optinal) Data preparation
To make input data in Mask-RCNN, a python file below separates overrapping mask in a original mask image using erosion and dilation algorithm (keneral size 7x7).   
Also, it reduces image size by half (1024x1024) and changes format from tiff to jpg to reduce memory load
```bash
   cd MASK_RCNN_ROOT/samples/nucleus
   python3 make_patch(erosion-dliation).py
```

## Training/Testing  
```bash
   cd MASK_RCNN_ROOT/samples/nucleus
   python3 nucleus_training.py
   python3 nucleus_testing.py
```


## Current Result

![Alt Text](https://github.com/ai-lab-circle/Mask-RCNN-nucleus-segmentation/blob/master/results.png)


## TODO

separate image into several parts, segment indepentantly.


## Citation

```bash
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}

```
