# COCO Object detection 

## How to use

Our code is based on  [mmdetection](https://github.com/open-mmlab/mmdetection), please install `mmdetection==2.23.0`. DeBiFormer serve as the strong backbones for
Mask R-CNN. It's easy to apply DeBiFormer in other detectors provided by mmdetection based on our examples. More details can be seen in [[paper]](https://arxiv.org/abs/2207.05501).

#### Training
To train  Mask R-CNN with DeBiFormer-S backbone using 8 gpus, run:
```shell
cd detection/
PORT=29501 bash dist_train.sh configs/mask_rcnn_debi_small_1x.py 8
```
#### Evaluation
To evaluate Mask R-CNN with DeBiFormer-S backbone using 8 gpus, run:
```shell
cd detection/
PORT=29501 bash dist_test.sh configs/mask_rcnn_debi_small_1x.py ../checkpoints/mask_rcnn_1x_debi_small.pth 8 --eval bbox
```


## Results

| name | Pretrained Model | Method | Lr Schd | mAP_box | mAP_mask/AP_M | weight | config |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeBiFormer-S | IN1k | MaskRCNN | 1x | 47.5 | 42.5 | [weight](https://drive.google.com/drive/folders/1hiTS_Xq1EfCOpgIBLb08lKMS30SBslRt) |[config](./configs/mask_rcnn_debi_small_1x.py) |
| DeBiFormer-B | IN1k | MaskRCNN | 1x | 48.5 | 43.2 | [weight](https://drive.google.com/drive/folders/1hiTS_Xq1EfCOpgIBLb08lKMS30SBslRt) | [config](./configs/mask_rcnn_debi_base_1x.py) |
| DeBiFormer-S | IN1k | RetinaNet | 1x | 45.6 | 49.3  | [weight](https://drive.google.com/drive/folders/1hiTS_Xq1EfCOpgIBLb08lKMS30SBslRt) |[config](./configs/retinanet_debi_small_1x.py) |
| DeBiFormer-B | IN1k | RetinaNet | 1x | 47.1 | 51.1  | [weight](https://drive.google.com/drive/folders/1hiTS_Xq1EfCOpgIBLb08lKMS30SBslRt) |[config](./configs/retinanet_debi_base_1x.py) |


## Acknowledgment 

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, and [DAT](https://github.com/LeapLabTHU/DAT), [BiFormer](https://github.com/rayleizhu/BiFormer) repositories.
