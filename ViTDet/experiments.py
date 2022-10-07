import sys
sys.path.insert(0, './detectron2/')

import torch
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tasks.datasets import COCOImages
from inference.experiment import Experiments 

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.data import MetadataCatalog

setup = {
    'device': 'cuda',
    'model_debug': False,
    'presaved_model': False,
    'train': True,
    'save_model': True,
    'save_location': '/scratch/bes1g19/UG-Research/VITT/saved_model/ViTT.pth'
}
hyperparameters = {
    'num_epochs': 30,
    'batch_size': 1,
    'image_resize': 128,
    'num_objects': 15,
    'accumulated_batches': 1,
    'num_workers': 8,
    'learning_rate': 1e-4,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'gamma': 0.1
}

def experiment(cfg):
    # put data into DataLoader
    train_imgs = COCOImages(train=True)
    val_imgs = COCOImages(train=False)

    train_dataloader = DataLoader(
        train_imgs,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        num_workers=hyperparameters['num_workers'],
    )
    val_dataloader = DataLoader(
        val_imgs,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        num_workers=hyperparameters['num_workers'],
    )

    # load model
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()
    print("\n\n\n")

    experiment = Experiments(model)
    experiment(train_dataloader)

if __name__ == '__main__':
    cfg = LazyConfig.load("./detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py")
    cfg.train.init_checkpoint = './saved_models/model_final_61ccd1.pkl'

    experiment(cfg)
