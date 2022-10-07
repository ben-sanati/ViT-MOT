import numpy as np
import torch
import torch.nn as nn
from typing import List

from detectron2.structures import Instances
from detectron2.utils.events import EventStorage

class Experiments(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, dataloader):
        with torch.no_grad():
            with EventStorage() as storage:
                for batch_index, (images) in enumerate(dataloader):
                    images = images.squeeze(0)
                    outputs = self.model([{'image': images}])
                    print(outputs)
                    """
                    # output bboxes are in xyxy format
                    pred_bboxes = [bbox for bbox in output[0]['instances'].get_fields()['pred_boxes'].tensor.tolist()]
                    pred_bboxes = torch.Tensor([list(bbox) for bbox in np.around(np.array(pred_bboxes),2)]).cuda()

                    # get bbox score
                    scores = output[0]['instances'].get_fields()['scores']
                    """
                    del images
                    del outputs
                    break

    def get_inputs(self, batch):
        return [
            {'image': image} for image in batch
        ]
