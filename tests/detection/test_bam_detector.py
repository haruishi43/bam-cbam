#!/usr/bin/env python3

from detectron2.config import get_cfg
from detectron2.modeling import build_model

from cbam.detectors.bam_backbone import build_bam_resnet_backbone


def test_loading_model():

    cfg = get_cfg()
    cfg.MODEL.BACKBONE.defrost()
    cfg.MODEL.BACKBONE.NAME = "build_bam_resnet_backbone"
    cfg.MODEL.BACKBONE.freeze()

    model = build_model(cfg)

    print(model)
