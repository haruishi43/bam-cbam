#!/usr/bin/env python3

import torch

from cbam.modules.cbam import (
    BasicConv,
    ChannelGate,
    ChannelPool,
    SpatialGate,
)


def test_basic_conv():
    r"""BasicConv Test"""
    channels = 4
    in_planes = channels
    out_planes = channels
    kernel_size = 3

    conv = BasicConv(
        in_planes=in_planes,
        out_planes=out_planes,
        kernel_size=kernel_size,
    )

    inp = torch.rand((1, channels, 16, 16), dtype=torch.float)

    out = conv(inp)
    assert inp.dim() == out.dim()
    assert out.shape[-3] == channels
    # NOTE: what else?


def test_channel_pool():
    r"""Channel Pooling Test"""
    channel_pool = ChannelPool()
    inp = torch.rand((1, 3, 16, 16), dtype=torch.float)
    out = channel_pool(inp)

    assert inp.dim() == out.dim()
    # takes max and mean of channel and concat, so channel becomes 2
    assert out.shape[-3] == 2


def test_spatial_gate():
    r"""Spatial Gate Test"""

    spatial_gate = SpatialGate(kernel_size=7)
    channels = spatial_gate.in_planes

    inp = torch.rand((1, channels, 16, 16), dtype=torch.float)
    out = spatial_gate(inp)
    assert inp.shape == out.shape, "input and output shape mismatch"


def test_channel_gate():
    r"""Channel Gate Test"""
    gate_channels = 64

    pool_types_combinations = [
        ["avg"],
        ["avg", "max"],
        ["avg", "max", "lp"],
        ["max"],
        ["max", "lp"],
        ["lse"],
    ]

    for pool_types in pool_types_combinations:
        channel_gate = ChannelGate(
            gate_channels=gate_channels,
            reduction_ratio=16,
            pool_types=pool_types,
        )
        inp = torch.rand((1, gate_channels, 16, 16), dtype=torch.float)
        out = channel_gate(inp)
        assert inp.shape == out.shape
        # NOTE: What other tests?
