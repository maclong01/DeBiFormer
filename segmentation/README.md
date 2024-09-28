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
```


#### Results

| name | Pretrained Model | Method | Crop Size | Lr Schd | mIoU | mIoU (ms + flip) | config |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[ DeBiFormer-B](https://drive.google.com/drive/folders/1Rko-MtpX52t5zZSwcQUfFrAvJIlMDHUl) | IN1k | S-FPN | 512x512 | 80K |50.6 | -  |[config](./configs/fpn_512_debi_base_80k.py)|
|[ DeBiFormer-B](https://drive.google.com/drive/folders/1Rko-MtpX52t5zZSwcQUfFrAvJIlMDHUl) | IN1k | UPerNet | 512x512 | 160K | 51.4 | 52.0 |[config](./configs/upernet_512_debi_base_160k.py) |

#### Acknowledgment 

This code is built using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [timm](https://github.com/rwightman/pytorch-image-models) libraries, [DAT](https://github.com/LeapLabTHU/DAT), and [BiFormer](https://github.com/rayleizhu/BiFormer) repositories.
