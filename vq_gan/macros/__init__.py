from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def Convolution1x1(input_channels: int, output_channels: int) -> nn.Module:
    
    return nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=1,
        stride=1,
        padding=0,
    )


def Convolution3x3(input_channels: int, output_channels: int) -> nn.Module:
    
    return nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    )


def Convolution4x4(input_channels: int, output_channels: int) -> nn.Module:

    return nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
    )


def Normalization(channels: int) -> nn.Module:

    return nn.GroupNorm(
        num_groups=min(channels, 32),
        num_channels=channels,
    )


def Repeat(module, channels_list: List[int]) -> nn.Module:

    return nn.Sequential(*(
        module(
            input_channels=input_channels, 
            output_channels=output_channels,
        ) for input_channels, output_channels in zip(
            channels_list[: -1], 
            channels_list[1 :],
        )
    ))
