# [DeBiFormer: Vision Transformer with Deformable Agent Bi-level Routing Attention]()

Official PyTorch implementation of **DeBiFormer**, from the following paper:

[DeBiFormer: Vision Transformer with Deformable Agent Bi-level Routing Attention](). ACCV 2024.\
[Nguyen Huu Bao Long](https://github.com/maclong01), [Chenyu Zhang](https://github.com/il1um), Yuzhi Shi, [Tsubasa Hirakawa](https://thirakawa.github.io/), [Takayoshi Yamashita](https://scholar.google.co.jp/citations?user=hkguTPgAAAAJ&hl=en), [Hironobu Fujiyoshi](https://scholar.google.com/citations?user=CIHKZpEAAAAJ&hl=en), and [Tohgoroh Matsui](https://xn--p8ja5bwe1i.jp/profile.html)

--- 
<p align="left">
<img src="assets/all_attention_3_1_v2-1.png" width=60% height=60% 
class="center">
</p>


<!-- ✅ ⬜️  -->

## News

* 2024-09-21: The paper has been accepted at ACCV 2024 !!!

## Results and Pre-trained Models

### ImageNet-1K trained models

| name | resolution |acc@1 | #params | FLOPs | model | log | tensorboard log<sup>*</sup> |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:| :---:| 
| DeBiFormer-T | 224x224 | 81.9 | 21.4 M | 2.6 G | [model](https://drive.google.com/drive/folders/1K_Zk5Etx2oh3yVccr71m1R3bTqWAI2bg) | [log](https://drive.google.com/drive/folders/1K_Zk5Etx2oh3yVccr71m1R3bTqWAI2bg) |
| DeBiFormer-S | 224x224 | 83.9 | 44 M | 5.4 G | [model](https://drive.google.com/drive/folders/1OmWKob1ECHgVMs5wSvZs3XF665zFJdHg) | [log](https://drive.google.com/drive/folders/1OmWKob1ECHgVMs5wSvZs3XF665zFJdHg) |
| DeBiFormer-B | 224x224 | 84.4 | 77 M | 11.8 G | [model](https://drive.google.com/drive/folders/1Ae3l2Q9nPbpOgSiTtX_HWyvSXIPQ9jce) | [log](https://drive.google.com/drive/folders/1Ae3l2Q9nPbpOgSiTtX_HWyvSXIPQ9jce) | 



## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Evaluation

We did evaluation on a slurm cluster environment, using the command below:

```bash
python hydra_main.py \
    data_path=./data/in1k input_size=224  batch_size=128 dist_eval=true \
    +slurm=${CLUSTER_ID} slurm.nodes=1 slurm.ngpus=8 \
    eval=true load_release=true model='biformer_small'
```

To test on a local machine, you may try

```bash
python -m torch.distributed.launch --nproc_per_node=8 main.py \
  --data_path ./data/in1k --input_size 224 --batch_size 128 --dist_eval \
  --eval --load_release --model biformer_small
```

This should give 
```
* Acc@1 83.754 Acc@5 96.638 loss 0.869
Accuracy of the network on the 50000 test images: 83.8%
```

**Note**: By setting `load_release=true`, the released checkpoints will be automatically downloaded, so you do not need to download manually in advance.

## Training

To launch training on a slurm cluster, use the command below:

```bash
python hydra_main.py \
    data_path=./data/in1k input_size=224  batch_size=128 dist_eval=true \
    +slurm=${CLUSTER_ID} slurm.nodes=1 slurm.ngpus=8 \
    model='biformer_small'  drop_path=0.15 lr=5e-4
```

**Note**: Our codebase automatically generates output directory for experiment logs and checkpoints, according to the passed arguments. For example, the command above will produce an output directory like

```
$ tree -L 3 outputs/ 
outputs/
└── cls
    └── batch_size.128-drop_path.0.15-input_size.224-lr.5e-4-model.biformer_small-slurm.ngpus.8-slurm.nodes.2
        └── 20230307-21:33:26
```

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, and [ConvNext](https://github.com/facebookresearch/ConvNeXt), [UniFormer](https://github.com/Sense-X/UniFormer) repositories.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```bibtex
@Article{zhu2023biformer,
  author  = {Lei Zhu and Xinjiang Wang and Zhanghan Ke and Wayne Zhang and Rynson Lau},
  title   = {BiFormer: Vision Transformer with Bi-Level Routing Attention},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2023},
}
```

## TODOs
- [x] Add camera-ready paper link
- [x] IN1k standard training code, log, and pretrained checkpoints
- [ ] IN1k token-labeling code
- [x] Semantic segmentation code
- [x] Object detection code
- [x] Swin-Tiny-Layout (STL) models
- [x] Refactor BRA and BiFormer code
- [ ] Visualization demo 
- [x] ~~More efficient implementation with triton~~. See [triton issue #1279](https://github.com/openai/triton/issues/1279)
- [ ] More efficient implementation (fusing gather and attention) with CUDA
