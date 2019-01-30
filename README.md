# Mask R-CNN for nucleus Detection and Segmentation (in progress)

This source includes nucleui segmentation code using Mask-RCNN.
Experiment was
Source coed was implemented on ubuntu 16.04, python3.5, CUDA9.0, and tensorflow1.12.0.

## Dataset

Download [pre-built images] for segmentation (https://drive.google.com/file/d/1uF_hxZJZTh8eWSwYuCP8gMGQw3ZxYEwD/view?usp=sharing)   
```bash
   mv downloaded_images  MASK_RCNN_ROOT/datasets/nuclues
```

(optional) Download [original nucleus dataset](http://andrewjanowczyk.com/wp-static/nuclei.tgz)


## (optinal) Data preparation
This function separates overrapping mask in a mask image.   
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
