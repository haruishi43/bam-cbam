#!/usr/bin/env python3

import os
import shutil

import torch


def save_checkpoint(state, is_best, prefix, default_path="./checkpoints"):
    if not os.path.exists(default_path):
        os.mkdir(default_path)
    filename = "{}/{}_checkpoint.pth.tar".format(default_path, prefix)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename, "{}/{}_model_best.pth.tar".format(default_path, prefix)
        )


def load_checkpoint(resume_path):
    assert os.path.exists(resume_path), "nothing at path {}".format(
        resume_path
    )
    checkpoint = torch.load(resume_path)
    return checkpoint
