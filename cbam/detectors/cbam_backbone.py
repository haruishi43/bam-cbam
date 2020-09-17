#!/usr/bin/env python3

from torch import nn

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


@BACKBONE_REGISTRY.register()
class ToyBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        # create your own backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=16, padding=3)

    def forward(self, image):
        return {"conv1": self.conv1(image)}

    def output_shape(self):
        return {"conv1": ShapeSpec(channels=64, stride=16)}
