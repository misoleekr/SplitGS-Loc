# [NeurIPS 2026] Disambiguating 2D-3D Correspondences in Gaussian Splatting-based Feature Fields for Visual Localization

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://misoleekr.github.io/SplitGS-Loc)
[![arXiv](https://img.shields.io/badge/arXiv-2605.07351-b31b1b.svg)](https://arxiv.org/abs/2605.07351)

This is the official pytorch implementation of SplitGS-Loc: [Disambiguating 2D-3D Correspondences in Gaussian Splatting-based Feature Fields for Visual Localization](https://arxiv.org/abs/2605.07351).

Authors: [Miso Lee](https://misoleekr.github.io), [Sangeek Hyun](https://hse1032.github.io), [Yerim Jeon](https://jyerim.github.io), [Jae-Pil Heo](https://sites.google.com/site/jaepilheo)


## Requirements
- CUDA 12.1
- Python 3.10
- Pytorch 2.4.1


## Installation

```bash
git clone --recursive https://github.com/misoleekr/SplitGS-Loc.git
cd SplitGS-Loc
conda create -n split python=3.10 -y
conda activate split
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
sh scripts/preprocess/install_mask2former.sh

cd submodules/simple-knn
python setup.py install
cd ../gsplat
cp ../../mygsplat/rendering.py gsplat/rendering.py
cp ../../mygsplat/_wrapper.py gsplat/cuda/_wrapper.py
cp ../../mygsplat/bindings.h gsplat/cuda/csrc/bindings.h
cp ../../mygsplat/rasterize_to_pixels_fwd.cu gsplat/cuda/csrc/rasterize_to_pixels_fwd.cu
python setup.py install
cd ../..
```

## Downloads
- Datasets:
    [CambridgeLandmarks](https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3#dataset) / [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- 7Scenes SfM poses:
    [visloc_pseudo_gt_limitations](https://github.com/tsattler/visloc_pseudo_gt_limitations#full-reconstructions) (`sfm_gt` → `sparse/0`)
- SuperPoint weights:
    [superpoint_v1.pth](https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth) → `encoders/sp_encoder/weights/superpoint_v1.pth`
- Pretrained RGB 3DGS / Split Feature Fields:
    [Hugging Face](https://huggingface.co/misoleekr/SplitGS-Loc)
```bash
pip install -U huggingface_hub
hf download misoleekr/SplitGS-Loc --include "3dgs/*" --local-dir datasets
```

## Data Preprocessing
```shell
sh scripts/preprocess/dataset_preprocess.sh

SplitGS-Loc
├── datasets
│   ├── 7scenes
│   │   ├── chess
│   │   │   ├── seq-01
│   │   │   ├── ...
│   │   │   ├── sparse/0
│   │   │   ├── TrainSplit.txt
│   │   │   └── TestSplit.txt
│   │   └── ...
│   ├── cambridge
│   │   ├── KingsCollege
│   │   │   ├── seq1
│   │   │   ├── ...
│   │   │   ├── reconstruction.nvm
│   │   │   ├── dataset_train.txt
│   │   │   └── dataset_test.txt
│   │   └── ...
│   └── 3dgs
│       ├── 7scenes
│       │   ├── chess
│       │   │   ├── cfg_args.json
│       │   │   ├── cameras.json
│       │   │   └── point_cloud/iteration_30000/point_cloud.ply
│       │   └── ...
│       └── cambridge
│           └── ...
└── results
    └── ${EXP_NAME}
        ├── chess
        │   ├── landmark_gs.ply
        │   ├── summary.json
        │   └── ...
        ├── ...
        ├── recall.txt
        └── median.txt
```

## Quick Evaluation
```shell
hf download misoleekr/SplitGS-Loc --include "results/*" --local-dir .
sh scripts/test_7scenes.sh 7scenes ${GPU_NUM}
sh scripts/test_cambridge.sh cambridge ${GPU_NUM}
```

## Running
```shell
# Training 3DGS
sh scripts/preprocess/train_7scenes.sh
sh scripts/preprocess/train_cambridge.sh

# Constructing Feature Fields
sh scripts/split_7scenes.sh ${EXP_NAME} ${GPU_NUM}
sh scripts/split_cambridge.sh ${EXP_NAME} ${GPU_NUM}

# Evaluation
sh scripts/test_7scenes.sh ${EXP_NAME} ${GPU_NUM}
sh scripts/test_cambridge.sh ${EXP_NAME} ${GPU_NUM}
```

## Citation
If our work is useful, please consider the following citation:
```
@article{lee2026splitgsloc,
  title={Disambiguating 2D-3D Correspondences in Gaussian Splatting-based Feature Fields for Visual Localization},
  author={Lee, Miso and Hyun, Sangeek and Jeon, Yerim and Heo, Jae-Pil},
  journal={arXiv preprint arXiv:2605.07351},
  year={2026}
}
```

## Acknowledgement
This repository is built based on [STDLoc](https://github.com/zju3dv/STDLoc) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) repositories.
Thank you for the great work.


## License
This project is released under the [Gaussian-Splatting License](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md).
