# Improving ViT Efficiency for MOT: Dynamic Tokenization

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) &emsp;
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) &emsp;
![Visual Studio](https://img.shields.io/badge/Visual%20Studio-5C2D91.svg?style=for-the-badge&logo=visual-studio&logoColor=white) &emsp;
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green) &emsp;
![LaTeX](https://img.shields.io/badge/latex-%23008080.svg?style=for-the-badge&logo=latex&logoColor=white) &emsp;
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

<br />

## Motivation 

Transformers have recently begun to dominate the computer vision (CV) landscape. One of the most researched CV tasks is object detection, a task where many object detection transformer varieties dominate the leaderboards. A major issue with transformers is the computational requirements of the model. Transformers are often very large models meaning large computational resources are required during inference. When it comes to vision transformers (ViTs), [1] find that the attention is sparse, meaning many of the tokens used in the model are not required and can be pruned. If achieved, we will aim to implement a ViT-based object detector and deploy the model to a video feed. The performance of the model on the video feed will then be investigated.


## Achievements

Given the time frame, this was a very ambitious project that accomplished many of its objectives and created the groundwork for future novel work to be completed. A plethora of material had to be learned to implement the tracker, from object detection and its metrics, transformers, ViTs, and its variants for object detection. 

Subsequently: 
- An efficient video loader that utilized sparse temporal sampling was implemented for the MOT17 dataset

<p align="center">
  <img src="/ViTDet/output_media/sparse_temporal_sampling.png" width="300"/>
</p>

- A backbone ViT called ViTDet was trained and implemented on the MOT17 and COCO datasets resulting in a high-quality object detector for video
- The ViTDet module was evaluated on video input feeds

<p align="center">
  <img src="/ViTDet/output_media/MOT17_example.gif" width="300"/>
</p>

- An A-ViT [1] module was implemented into a normal ViT for classification (not enough time to apply to ViTDet module)

## Future Work

An AViT module from [2] was implemented into a normal ViT for classification, however, there was not sufficient time for implementation and training of the module into the ViTDet object detector. For future work, the A-ViT module will be implemented into the ViTDet object detector and then modified such that the module prunes redundant tokens based on frame-to-frame spatial similarity as well as spatial image complexity. Once this is completed, the object detector can then be implemented with Deep-SORT for improved tracking performance.

## Run

It should be noted that this project is **NOT** maintained. 

0. Requirements:
    * python3.7, pytorch 1.11, torchvision 0.12.0, cuda 11.3, cudnn 8.2.0
    * Prepare MOT17 and MS-COCO dataset
1. Model Use:
    * The pre-trained ViTDet model can be attained from ![detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet) (I used the COCO mask R-CNN ViT-B version)
    * The model can be experimented on MOT17 using the [experiments file](/ViTDet/experiments.py) 


## Final Report 

The final report for the project can be found [here](/UG_Research_Report_Final.pdf), and the presentation given in the project viva can be found [here](/Presentation.pptx).

### References 

[1] H. Yin, A. Vahdat, J. M. Alvarez, A. Mallya, J. Kautz, and P. Molchanov, “A-vit: Adaptive tokens for efficient vision transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp.10 809–10 818. 
