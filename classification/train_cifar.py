#!/usr/bin/env python3

import argparse
import random

from PIL import ImageFile

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from cbam.cifar import create_resnet

from classification.utils import (
    adjust_learning_rate,
    load_checkpoint,
    save_checkpoint,
    train,
    validate,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
    parser.add_argument("--data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        metavar="PFX",
        help="prefix for logging & checkpoint saving",
    )
    parser.add_argument(
        "--att-type",
        type=str,
        choices=["BAM", "CBAM"],
        default=None,
    )
    parser.add_argument(
        "--depth",
        default=50,
        type=int,
        metavar="D",
        help="model depth",
    )
    parser.add_argument(
        "--ngpu",
        default=4,
        type=int,
        metavar="G",
        help="number of gpus to use",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum",
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        metavar="BS",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluation only",
    )
    parser.add_argument(
        "--cifar100",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    # create model
    model = create_resnet(args.depth, 1000, args.att_type)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    print("model")
    print(model)

    # get the number of model parameters
    print(
        "Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )

    # optionally resume from a checkpoint
    best_prec1 = 0
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = load_checkpoint(args.resume)
        args.start_epoch = checkpoint["epoch"]
        best_prec1 = checkpoint["best_prec1"]
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
        )

    # Data loading code
    root = args.data
    img_size = 32
    normalize = transforms.Normalize(
        mean=[0.414, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )

    dataset_cls = datasets.CIFAR10 if not args.cifar100 else datasets.CIFAR100
    val_loader = torch.utils.data.DataLoader(
        dataset_cls(
            root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(40),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            download=True,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    if args.evaluate:
        validate(val_loader, model, criterion, 0, 1)
        return

    train_dataset = dataset_cls(
        root,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(
            train_loader, model, criterion, optimizer, epoch, args.print_freq
        )

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, args.print_freq)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args.prefix,
        )


if __name__ == "__main__":
    main()
