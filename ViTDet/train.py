import sys
sys.path.insert(0, '/scratch/bes1g19/UG-Research/ViTDet/detectron2/')
import torch
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model

cfg = LazyConfig.load("/scratch/bes1g19/UG-Research/ViTDet/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py")
cfg.train.init_checkpoint = '/scratch/bes1g19/UG-Research/ViTDet/saved_models/model_final_61ccd1.pkl' # replace with the path were you have your model
classes = metadata.thing_classes

model = instantiate(cfg.model).cuda()
model = create_ddp_model(model)
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

print(model)
