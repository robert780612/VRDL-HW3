# VRDL-HW3
Code for Selected Topics in Visual Recognition using Deep Learning Homework 3

## Briefly Introduction
In this homework assignment, we need to train a 20-categories instance segmentation model using only 1349 images sampled from Pascal VOC. I adopt the notable algorithm [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) to solve this task, Mask R-CNN only adds a small overhead to Faster R-CNN. With 270000 training iterations, I got mAP 0.47097 on the test dataset.

## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
- NVIDIA GTX 1080ti

## Reproducing Submission
To reproduct my submission, do the following steps:
1. [Requirement](#requirement)
2. [Dataset Preparation](#dataset-preparation)
3. [Configuration](#configuration)
4. [Training](#training)
5. [Inference](#inference)

## Requirement
The requirements are listed as below
- Python >= 3.6
- numpy
- tqdm
- torch
- torchvision
- tensorboard
- h5py
- pandas 
- matplotlib
- cv2
- pillow
- detectron2
- pycocotools 
- skimage 

## Dataset Preparation
### Download Official Image
Download and extract Pascal VOC dataset. Put them into "./pascal voc"

### Prepare Images
Detectron2 supports standard COCO and Pascal VOC format, so there is no need any preprocessing.

## Configuration
Set the configuration in the *config.py* (Model config, Dataset config...). You can find the detail explaination in this [website](https://detectron2.readthedocs.io/modules/config.html#config-references).

## Training
To train models, run following commands. All training log and trained model is saved in "output" directory.
```
$ python train.py
```

## Inference
If trained weights are prepared, you can run the following command to generate json file which contains predicted results. The final detection performance on the test dataset is mAP 0.47097.
```
$ python eval.py
```
And if you set ```visualize``` to ```True```. It will plot predicted box on the image and save it.

## Reference
- This repository is based on [Detectron2](https://github.com/facebookresearch/detectron2).
