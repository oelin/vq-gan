from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from vq_gan.macros import (
    Convolution1x1, 
    Convolution3x3, 
    Convolution4x4,
    Normalization,
    Repeat,
)


class UpsampleBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.normalization = Normalization(channels=input_channels)
        
        self.convolution = Convolution3x3(
            input_channels=input_channels, 
            output_channels=output_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.normalization(x)
        x = F.leaky_relu(x)
        x = self.upsample(x)
        x = self.convolution(x)

        return x


class DownsampleBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.normalization = Normalization(channels=input_channels)

        self.convolution = Convolution4x4(
            input_channels=input_channels,
            output_channels=output_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.normalization(x)
        x = F.leaky_relu(x)
        x = self.convolution(x)

        return x


class ResidualBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.normalization = Normalization(channels=channels)

        self.convolution = Convolution3x3(
            input_channels=channels,
            output_channels=channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = self.normalization(x)
        z = F.leaky_relu(z)
        z = self.convolution(z)

        return x + z


class ResNetBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.residual_block_1 = ResidualBlock(channels=channels)
        self.residual_block_2 = ResidualBlock(channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.residual_block_1(x)
        x = self.residual_block_2(x)

        return x
  

class AttentionBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.normalization = Normalization(channels=channels)

        self.convolution_1 = Convolution1x1(
            input_channels=channels,
            output_channels=channels,
        )

        self.convolution_2 = Convolution1x1(
            input_channels=channels,
            output_channels=channels,
        )

        self.convolution_3 = Convolution1x1(
            input_channels=channels,
            output_channels=channels,
        )

        self.convolution_4 = Convolution1x1(
            input_channels=channels,
            output_channels=channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape

        z = self.normalization(x)
        q = self.convolution_1(z)
        k = self.convolution_2(z)
        v = self.convolution_3(z)

        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')  # Transposed.
        v = rearrange(v, 'b c h w -> b (h w) c')

        z = F.softmax(q @ k, dim=-1) @ v
        z = rearrange(z, 'b (h w) c -> b c h w', h=H, w=W)
        z = self.convolution_4(z)

        return x + z


class DownBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) ->  None:
        super().__init__()

        self.resnet_block = ResNetBlock(channels=input_channels)
        
        self.downsample_block = DownsampleBlock(
            input_channels=input_channels,
            output_channels=output_channels,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.resnet_block(x)
        x = self.downsample_block(x)

        return x


class UpBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.resnet_block = ResNetBlock(channels=input_channels)

        self.upsample_block = UpsampleBlock(
            input_channels=input_channels,
            output_channels=output_channels,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.resnet_block(x)
        x = self.upsample_block(x)

        return x


class MiddleBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.resnet_block_1 = ResNetBlock(channels=channels)
        self.resnet_block_2 = ResNetBlock(channels=channels)
        self.attention_block = AttentionBlock(channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.resnet_block_1(x)
        x = self.attention_block(x)
        x = self.resnet_block_2(x)

        return x


class Encoder(nn.Module):

    def __init__(self, channels_list: List[int]) -> None:
        super().__init__()

        self.down_blocks = Repeat(module=DownBlock, channels_list=channels_list)
        self.middle_block = MiddleBlock(channels=channels_list[-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.down_blocks(x)
        x = self.middle_block(x)

        return x


class Decoder(nn.Module):
    
    def __init__(self, channels_list: List[int]) -> None:
        super().__init__()

        self.up_blocks = Repeat(module=UpBlock, channels_list=channels_list)
        self.middle_block = MiddleBlock(channels=channels_list[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.middle_block(x)
        x = self.up_blocks(x)

        return x


class GaussianDistribution(nn.Module):

    def __init__(self, parameters: torch.Tensor) -> None:
        super().__init__()

        self.mean, self.log_variance = parameters.chunk(chunks=2, dim=1)
    
    def sample(self) -> torch.Tensor:

        epsilon = torch.randn_like(self.mean, device=self.mean.device)
        standard_deviation = torch.exp(0.5 * self.log_variance)
        x = epsilon * standard_deviation + self.mean

        return x
