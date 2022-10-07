from pycocotools.coco import COCO
import torch
import torchvision.transforms as transforms
from tasks.video_dataset import VideoFrameDataset, ImglistToTensor
from tasks.bbox_ops import BBoxOperations
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

"""
    CocoDetection outputs:
        Image of dimension [N x C x H x W]
        Labels of dimensions [N x num_objects x (1+4)]
            The labels are in xywh form and the bbox values are normalized to be between 0 and 1
"""
class CocoDetection(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, image_size, num_objects, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.num_objects = num_objects

        self.bbox_ops = BBoxOperations(image_size)
        self.image_size = image_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        width, height = img.size
        if self.transform is not None:
            img = self.transform(img)

        # target albumations
        bboxes = []
        for idx in range(len(target)):
            bbox_info = target[idx]['bbox']
            # convert from xywh to xyxy
            bbox_info = self.bbox_ops.xywh_to_xyxy(torch.Tensor(bbox_info).reshape(1, 1, 4))

            # normalize bbox
            bbox_info = bbox_info / self.image_size

            # transform the bbox to the image size
            bbox_info = self.resize_bbox(bbox_info, width, height)

            # convert back to xywh
            bbox_info = self.bbox_ops.xyxy_to_xywh(torch.Tensor(bbox_info).reshape(1, 1, 4))

            bbox_info = bbox_info.reshape(-1).tolist()
            bbox_info.insert(0, 1)
            bboxes.append(bbox_info)

        while len(bboxes) < self.num_objects:
            bboxes.append([0, 0, 0, 0, 0])

        while len(bboxes) > self.num_objects:
            bboxes.pop(-1)

        bboxes = torch.tensor(bboxes)

        img = img.float()
        bboxes = bboxes.float()
    
        return img, bboxes


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def resize_bbox(self, bbox, width, height):
        x_scale = (self.image_size / width)
        y_scale = (self.image_size / height)

        bbox[..., 0] = bbox[..., 0] * x_scale
        bbox[..., 1] = bbox[..., 1] * y_scale
        bbox[..., 2] = bbox[..., 2] * x_scale
        bbox[..., 3] = bbox[..., 3] * y_scale

        return bbox


class COCODatasetObjectDetection:
    def __init__(self, num_objects, image_size, batch_size, num_workers):
        root = '/ECShome/ECSdata/data_sets/coco_2017/'
        self.train_root = root + 'train2017/'
        self.train_annotations_file = root + 'annotations/instances_train2017.json'
        self.val_root = root + 'val2017/'
        self.val_annotations_file = root + 'annotations/instances_val2017.json'

        self.num_objects = num_objects
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([image_size, image_size]),
            transforms.Normalize(mean=mean.tolist(),
                                 std=std.tolist())
        ])

        #self.unnormal = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        self.unnormal = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                                 std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                                 std = [ 1., 1., 1. ]),
            ])

        self.enlarge_val = 512
        self.enlarge = transforms.Compose([transforms.Resize([self.enlarge_val, self.enlarge_val])])
    
    def get_data(self):
        train_dataset = CocoDetection(
            root=self.train_root,
            annFile=self.train_annotations_file,
            image_size=self.image_size,
            num_objects=self.num_objects,
            transform=self.transform
        )
        val_dataset = CocoDetection(
            root=self.val_root,
            annFile=self.val_annotations_file,
            image_size=self.image_size,
            num_objects=self.num_objects,
            transform=self.transform
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_dataloader, val_dataloader

    def UnnormImg(self, image):
        image = self.unnormal(image)

        return image

    def EnlargeImg(self, image, bbox):
        bbox_size = CocoDetection(None, None, self.enlarge_val, None)
        image = self.enlarge(image)
        bbox = bbox_size.resize_bbox(bbox, self.image_size, self.image_size)

        return image, bbox


class COCOImages(Dataset):
    def __init__(self, train):
        super().__init__()

        self.train_root = '/ECShome/ECSdata/data_sets/coco_2017/train2017/'
        self.val_root = '/ECShome/ECSdata/data_sets/coco_2017/val2017/'
        self.train = train
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256])
        ])

        train_annotations = '/ECShome/ECSdata/data_sets/coco_2017/annotations/instances_train2017.json'
        val_annotations = '/ECShome/ECSdata/data_sets/coco_2017/annotations/instances_val2017.json'

        self.coco_train = COCO(train_annotations)
        self.coco_val = COCO(val_annotations)
        self.ids_train = list(self.coco_train.imgs.keys())
        self.ids_val = list(self.coco_val.imgs.keys())

    def __getitem__(self, index):
        coco_train = self.coco_train
        coco_val = self.coco_val

        img_id_train = self.ids_train[index]
        img_id_val = self.ids_val[index]

        path_train = coco_train.loadImgs(img_id_train)[0]['file_name']
        path_val = coco_val.loadImgs(img_id_val)[0]['file_name']

        img_train = Image.open(os.path.join(self.train_root, path_train)).convert('RGB')
        img_val = Image.open(os.path.join(self.val_root, path_val)).convert('RGB')

        img_train = self.transform(img_train)
        img_val = self.transform(img_val)

        if self.train:
            return img_train.float()
        else: 
            return img_val.float()

    def __len__(self):
        if self.train:
            return len(self.ids_train)
        else:
            return len(self.ids_val)
