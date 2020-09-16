#!/usr/bin/env python3

import time

from .accuracy import accuracy
from .average_meter import AverageMeter


def validate(val_loader, model, criterion, epoch, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )

    print(
        " * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(
            top1=top1, top5=top5
        )
    )

    return top1.avg
