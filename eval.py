
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
from glob import glob

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import binary_mask_to_rle
from config import cfg


#  1349 trianing images, 100 testing images
visualize = True
infolder = './pascal voc/test_images'
outfolder = 'pred'
model_name = 'model_final.pth'
test_images = glob(os.path.join(infolder, '*.jpg'))
test_json = './pascal voc/test.json'


# Registry dataset
category_mapping = {
                'aeroplane': 0,
                'bicycle': 1, 
                'bird': 2, 
                'boat': 3, 
                'bottle': 4, 
                'bus': 5, 
                'car': 6,
                'cat': 7,
                'chair': 8,
                'cow': 9,
                'diningtable': 10,
                'dog': 11,
                'horse': 12,
                'motorbike': 13,
                'person': 14,
                'pottedplant': 15,
                'sheep': 16,
                'sofa': 17,
                'train': 18,
                'tvmonitor': 19}

MetadataCatalog.get("hw3_train").set(thing_classes = list(category_mapping.keys()),
                                        stuff_classes = list(category_mapping.keys()))


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)  # path to the model we just trained
cfg.TEST.AUG.FLIP = False
cfg.MODEL.DEVICE = 'cpu'
#
predictor = DefaultPredictor(cfg)
#
cocoGt = COCO(test_json)
predictions = []
for i, imgid in enumerate(cocoGt.imgs):
    print(i, imgid)
    image = cv2.imread(os.path.join(infolder, cocoGt.loadImgs(ids=imgid)[0]['file_name']))[:,:,::-1] # load image
    output = predictor(image) # run inference of your model
    for i in range(len(output['instances'].scores)): # Loop all instances
        category_id = output['instances'].pred_classes[i] + 1  # from 1 to 20
        score = output['instances'].scores[i]
        rle = binary_mask_to_rle(output['instances'].pred_masks[i].numpy())
        predictions.append({'image_id': imgid, 'category_id': int(category_id.numpy()), 'segmentation': rle, 'score': float(score.numpy())})

    if visualize:
        v = Visualizer(image, MetadataCatalog.get("hw3_train"), scale=1.2)
        out = v.draw_instance_predictions(output["instances"].to("cpu"))
        out_image = out.get_image()
        cv2.imwrite(os.path.join('pred', cocoGt.loadImgs(ids=imgid)[0]['file_name']), out_image[:,:,::-1] )

with open('0786039.json', 'w') as f:
    json.dump(predictions, f)


