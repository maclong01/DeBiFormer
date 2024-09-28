# COCO Object detection 

## How to use

Our code is based on  [mmdetection](https://github.com/open-mmlab/mmdetection), please install `mmdetection==2.23.0`. Next-ViT serve as the strong backbones for
Mask R-CNN. It's easy to apply Next-ViT in other detectors provided by mmdetection based on our examples. More details can be seen in [[paper]](https://arxiv.org/abs/2207.05501).

#### Training
To train  Mask R-CNN with Next-ViT-S backbone using 8 gpus, run:
```shell
cd detection/
PORT=29501 bash dist_train.sh configs/mask_rcnn_nextvit_small_1x.py 8
```
#### Evaluation
To evaluate Mask R-CNN with Next-ViT-S backbone using 8 gpus, run:
```shell
cd detection/
PORT=29501 bash dist_test.sh configs/mask_rcnn_nextvit_small_1x.py ../checkpoints/mask_rcnn_1x_nextvit_small.pth 8 --eval bbox
```


## Results

| name | Pretrained Model | Method | Lr Schd | mAP_box | mAP_mask | log | mAP_box<sup>*</sup> |  mAP_mask<sup>*</sup> | config |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeBiFormer-S | IN1k | MaskRCNN | 1x | 47.8 | 43.2 | [log]() | 48.1 | 43.6 | [config](./configs/coco/maskrcnn.1x.biformer_small.py) |
| DeBiFormer-B | IN1k | MaskRCNN | 1x | 48.6 | 43.7 | [log]() | - | - | [config](./configs/coco/maskrcnn.1x.biformer_base.py) |
| DeBiFormer-S | IN1k | RetinaNet | 1x | 45.9 | - | [log]() | 47.3 | - | [config](./configs/coco/retinanet.1x.biformer_small.py) |
| DeBiFormer-B | IN1k | RetinaNet | 1x | 47.1 | - | [log]() |- | - |[config](./configs/coco/retinanet.1x.biformer_base.py) |


## Acknowledgment 

This code is built using [mmdetection](https://github.com/open-mmlab/mmdetection), [timm](https://github.com/rwightman/pytorch-image-models) libraries, and [UniFormer](https://github.com/Sense-X/UniFormer) repository.
