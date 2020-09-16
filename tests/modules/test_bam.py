#!/usr/bin/env python3

import torch

from cbam.modules.bam import (
    ChannelGate,
    SpatialGate,
)


def test_channel_gate():
    r"""Channel Gate Test"""
    batch = 4
    gate_channel = 64
    channel_gate = ChannelGate(
        gate_channel=gate_channel,
        reduction_ratio=16,
        num_layers=1,
    )
    # print(channel_gate.gate_c)
    inp = torch.rand((batch, gate_channel, 16, 16), dtype=torch.float)
    out = channel_gate(inp)

    assert inp.shape == out.shape


def test_spatial_gate():
    r"""Spatial Gate Test"""
    batch = 4
    gate_channel = 64
    spatial_gate = SpatialGate(
        gate_channel=gate_channel,
        reduction_ratio=16,
    )
    # print(spatial_gate.gate_s)
    inp = torch.rand((batch, gate_channel, 16, 16), dtype=torch.float)
    out = spatial_gate(inp)

    assert inp.shape == out.shape
