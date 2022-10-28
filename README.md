# FSS-Codebase
This repo presents a scalable codebase for few-shot segmentation research. Some major references are <a href="https://github.com/dvlab-research/PFENet" target="_blank">**PFENet**</a>, <a href="https://github.com/zhiheLu/CWT-for-FSS" target="_blank">**CWT**</a>, and <a href="https://github.com/rstrudel/segmenter" target="_blank">**Segmenter**</a>.

## Requisites
- Test Env: Python 3.9.7 (Singularity)
    - Path on NYU Greene: `/scratch/hl3797/overlay-25GB-500K.ext3`
- Packages:
    - torch (1.10.2+cu113), torchvision (0.11.3+cu113), timm (0.5.4)
    - numpy, scipy, pandas, tensorboardX
    - cv2, einops

## Clone codebase
```shell
git clone https://github.com/hmdliu/FSS-Codebase && cd FSS-Codebase
```

## Preparation

### PASCAL-5i dataset
**Note:** Make sure the path in `scripts/prepare_pascal.sh` works for you.
```shell
# default data root: ../dataset/VOCdevkit/VOC2012
bash scripts/prepare_pascal.sh
```

### COCO-20i dataset
You may refer to <a href="https://github.com/dvlab-research/PFENet#datasets-and-data-preparation" target="_blank">PFENet</a> for more details.

### Pretrained models
For ImageNet pre-trained models, please download it <a href="https://drive.google.com/file/d/1rMPedZBKFXiWwRX3OHttvKuD1h9QRDbU/view?usp=sharing" target="_blank">here</a> (credits <a href="https://github.com/dvlab-research/PFENet" target="_blank">PFENet</a>) and unzip as `initmodel/`. For models pre-trained on the base classes, you may find it <a href="https://drive.google.com/file/d/1VPBquiy4DZXu8b6qsSB6XtO5_6jTpXgo/view?usp=sharing" target="_blank">here</a> (credits <a href="https://github.com/zhiheLu/CWT-for-FSS" target="_blank">CWT</a>) and rename them as follows: `pretrained/[dataset]/split[i]/pspnet_resnet[layers]/best.pth`.

## Dir explanations
- **initmodel**: ImageNet pre-trained backbone weights. `.pth`
- **pretrained**: Base classes pre-trained backbone weights. `.pth`
- **configs**: Experiment configurations. `.yaml`
- **results**: Experiment logs and checkpoints. `.log` `.pth` `.yaml`
- **scripts**: Training and helper scripts. `.sh` `.slurm`
- **src**: Source code. `.py`

## Sample Usage
**exp_id**: [exp_group]_[meta_cfg]_[train_cfg] (see `src/exp.py` for more details)
```shell
# debug mode (i.e., only log to shell)
python -m src.train --config configs/pascal_sample.yaml --exp_id sample_wd_pm11 --debug True

# submit to slurm
sbatch scripts/train_pascal.slurm configs/pascal_sample.yaml sample_wd_pm11

# output dir: results/sample/wd_pm11
tail results/sample/wd_pm11/output.log
```
