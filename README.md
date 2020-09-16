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

## Detection
