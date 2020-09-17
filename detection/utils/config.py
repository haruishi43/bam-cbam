#!/usr/bin/env python3

from detectron2.config import CfgNode as CN


def add_cbam_config(cfg) -> None:
    r"""Add config for CBAM"""

    cfg.MODEL.CBAM = CN()
