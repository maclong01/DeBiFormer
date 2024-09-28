# ADE20k Semantic segmentation

#### Training
To train Semantic FPN 80k with DeBiFormer-B backbone using 8 gpus, run:
```shell
cd segmentation/
PORT=29501 bash dist_train.sh configs/fpn_512_debi_base_80k.py 8
```


#### Evaluation
To evaluate Semantic FPN 80k(single scale) with DeBiFormer-B backbone using 8 gpus, run:
```shell
cd segmentation/
PORT=29501 bash dist_test.sh configs/fpn_512_debi_base_80k.py ../checkpoints/fpn_80k_debi_base.pth 8 --eval mIoU



## Results

| name | Pretrained Model | Method | Crop Size | Lr Schd | mIoU | mIoU (ms + flip) | config |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BiFormer-B | IN1k | S-FPN | 512x512 | 80K | 49.9 | -  | - | [config](./configs/ade20k/sfpn.biformer_base.py)|
| BiFormer-B | IN1k | UPerNet | 512x512 | 160K | 51.0 | 51.7 |  - | [config](./configs/ade20k/upernet.biformer_base.py) |

## Acknowledgment 

This code is built using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [timm](https://github.com/rwightman/pytorch-image-models) libraries, and [UniFormer](https://github.com/Sense-X/UniFormer) repository.
