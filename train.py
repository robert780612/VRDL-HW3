import os
import numpy as np
from PIL import Image

from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco import load_coco_json

from config import cfg

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

def load_train():
    hw3_train = load_coco_json('./pascal voc/pascal_train.json', './pascal voc/train_images', dataset_name='hw3_train')
    return hw3_train

DatasetCatalog.register("hw3_train", load_train)
MetadataCatalog.get("hw3_train").set(thing_classes = list(category_mapping.keys()),
                                        stuff_classes = list(category_mapping.keys()))

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

