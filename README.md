# BAM and CBAM

A fork of [BAM and CBAM](https://github.com/Jongchan/attention-module).
- update for newer version of PyTorch
- cleaned up code, organized directory
- debugged and updated training script
- Added training for other tasks
- development (`black`, `pytest`, etc)

## Prerequisites

- Python >= 3.6
- PyTorch >= 1.5
- [Detectron2](https://github.com/facebookresearch/detectron2) >= 0.21
- fvcore (comes with Detectron2)

If you don't have [detectron2](https://github.com/facebookresearch/detectron2), you can either install it yourself, or use:
```Bash
git clone --recursive git@github.com:haruishi43/bam-cbam.git
cd third/detectron2
pip install .
```

## Classification

### CIFAR10 and CIFAR100

```Bash
python classification/train_cifar.py --data <path/to/dataset/root> --prefix cifar_run_1
```

- Train CIFAR100 using `--cifar100`.
- See other arguments inside the `parse_args()` function @ [`train_cifar.py`](classification/train_cifar.py).

### ImageNet

```Bash
python classification/train_imagenet.py --data <path/to/dataset/root> --prefix imagenet_run_1
```

- See other arguments inside the `parse_args()` function @ [`train_imagenetpy`](classification/train_imagenet.py).
- Download ImageNet from the [official website](http://www.image-net.org/challenges/LSVRC/2012/downloads) and use this [script](https://gist.github.com/haruishi43/dc96e069ba4d32104ed9b1761f55c2ee) to orgaize it.

## Detection

### COCO

Orgaize COCO dataset ([see detectron2's guide for more information](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md)).

```Bash
export DETECTRON2_DATASETS=/path/to/datasets  # else detectron2 will use ./datasets
```

```Bash
python detection/train_coco.py --num-gpus 8 \
  --config-file detection/configs/COCO-Detection/faster_rcnn_R_50_CBAM_1x.yaml
```

The original configuration are for using 8 gpus, you might need to change parameters for single gpu:
```Bash
CUDA_VISIBLE_DEVICES=0, python detection/train_coco.py --ngpu 1 \
  --config-file detection/configs/COCO-Detection/faster_rcnn_R_50_CBAM_1x.yaml \
  SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

### Pascal VOC
