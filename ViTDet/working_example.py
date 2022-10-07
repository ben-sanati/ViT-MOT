import sys
sys.path.insert(0, './detectron2/')

import torch
import numpy as np
from PIL import Image
import cv2
import os
import torchvision.transforms as transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.data import MetadataCatalog

cfg = LazyConfig.load('./detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py')
cfg.train.init_checkpoint = './saved_models/model_final_61ccd1.pkl'
metadata = MetadataCatalog.get(cfg.dataloader.train.dataset.names)  # to get labels from ids
classes = metadata.thing_classes

model = instantiate(cfg.model)
model = create_ddp_model(model)
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

model.eval()

filename1 = 'C:/Users/benja\Documents/GitHub/UGResearch-Project/Prep/MOT17/train/MOT17-02-DPM/img1/000001.jpg'
filename2 = 'C:/Users/benja\Documents/GitHub/UGResearch-Project/Prep/MOT17/train/MOT17-04-SDP/img1/000001.jpg'
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([512, 512])
])

single_img = True
if single_img:
    image = Image.open(filename2)
    image = np.array(image, dtype=np.uint8)# the model expects the image to be in channel first format (C, H, W)
    with torch.inference_mode():
        image = transforms(image) * 255
        output = model([{'image': image}])
        # output bboxes are in xyxy format
        pred_bboxes = [bbox for bbox in output[0]['instances'].get_fields()['pred_boxes'].tensor.tolist()]
        pred_bboxes = torch.Tensor([list(bbox) for bbox in np.around(np.array(pred_bboxes), 2)]).cuda()

        pred_classes = output[0]['instances'].get_fields()['pred_classes']
        print(pred_classes, len(pred_classes.tolist()))

        # get bbox score
        scores = output[0]['instances'].get_fields()['scores']

        # visualize img
        image = image.numpy()
        image = np.moveaxis(image, 0, -1)  # CV2 expects the image to be in channel last format (H, W, C)
        image = np.ascontiguousarray(image, dtype=np.uint8)

        accum = 0
        for bbox, score in zip(pred_bboxes, scores):
            if score > 0.3:
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
                accum += 1

        print(accum)

        status = cv2.imwrite('./output_media/example.jpg', (image))
        print("Image written to file-system : ", status)

        # get mAP
        pred_classes[pred_classes != 0] = 1
        pred_bboxes = pred_bboxes.tolist()
        pred_classes = pred_classes.tolist()
        scores = scores.tolist()
        """
        preds = [
            dict(
                boxes=pred_bboxes,
                scores=scores,
                labels=pred_classes
            )
        ]
        print(preds)
        targets = [
            dict(
                boxes=,
                labels=
            )
        ]
        
        map = MeanAveragePrecision()
        map.update(preds, targets)
        val = map.compute()
        print(val['map'].item())
        """
else:
    image1 = Image.open(filename1)
    image2 = Image.open(filename2)
    image1 = np.array(image1, dtype=np.uint8)# the model expects the image to be in channel first format (C, H, W)
    image2 = np.array(image2, dtype=np.uint8)# the model expects the image to be in channel first format (C, H, W)
    with torch.inference_mode():
        image1 = transforms(image1) * 255
        image2 = transforms(image2) * 255
        images = [{'image': image} for image in (image1, image2)]
        output = model(images)

        pred_bboxes = []
        scores = []
        for out in output:
            bboxes = [bbox for bbox in out['instances'].get_fields()['pred_boxes'].tensor.tolist()]
            pred_bboxes.append(torch.Tensor([list(bbox) for bbox in np.around(np.array(bboxes), 2)]))
            scores.append(out['instances'].get_fields()['scores'])

    # visualize images
    image1 = image1.numpy()
    image1 = np.moveaxis(image1, 0, -1)  # CV2 expects the image to be in channel last format (H, W, C)
    image1 = np.ascontiguousarray(image1, dtype=np.uint8)

    image2 = image2.numpy()
    image2 = np.moveaxis(image2, 0, -1)  # CV2 expects the image to be in channel last format (H, W, C)
    image2 = np.ascontiguousarray(image2, dtype=np.uint8)

    accum1 = 0
    for bbox, score in zip(pred_bboxes[0], scores[0]):
        if score > 0.3:
            cv2.rectangle(image1, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
            accum1 += 1

    accum2 = 0
    for bbox, score in zip(pred_bboxes[1], scores[1]):
        if score > 0.3:
            cv2.rectangle(image2, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
            accum2 += 1

    print(f"# objects in 1: {accum1}\n# objects in 2: {accum2}")

    status1 = cv2.imwrite('./output_media/example1.jpg', (image1))
    status2 = cv2.imwrite('./output_media/example2.jpg', (image2))
    print("Image 1 written to file-system : ", status1)
    print("Image 2 written to file-system : ", status2)
