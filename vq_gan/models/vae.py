from typing import Tuple, List

from dataclasses import dataclass

from . macros import Convolution3x3
from . modules import Encoder, Decoder, GaussianDistribution


@dataclass(frozen=True)
class VAEOptions:

    input_channels: int
    output_channels: int
    latent_channels: int
    encoder_channels_list: List[int]
    decoder_channels_list: List[int]


class VAE(nn.Module):

    def __init__(self, options: VAEOptions) -> None:
        super().__init__()

        self.encoder = Encoder(channels_list=options.encoder_channels_list)
        self.decoder = Decoder(channels_list=options.decoder_channels_list)

        # Input to encoder.

        self.convolution_1 = Convolution3x3(
            input_channels=options.input_channels,
            output_channels=options.encoder_channels_list[0],
        )

        # Encoder to latent.

        self.convolution_2 = Convolution3x3(
            input_channels=options.encoder_channels_list[-1],
            output_channels=options.latent_channels * 2,
        )

        # Latent to decoder.

        self.convolution_3 = Convolution3x3(
            input_channels=options.latent_channels,
            output_channels=options.decoder_channels_list[0],
        )

        # Decoder to output.

        self.convolution_4 = Convolution3x3(
            input_channels=options.decoder_channels_list[-1],
            output_channels=options.output_channels,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:

        x = self.convolution_1(x)
        x = self.encoder(x)
        z = self.convolution_2(x)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:

        x = self.convolution_3(z)
        x = self.decoder(x)
        x = self.convolution_4(x)

        return x

    def forward(
        self, 
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, GaussianDistribution]:

        distribution = GaussianDistribution(self.encode(x))  # Posterior.
        z = distribution.sample()
        x = self.decode(z)
    
        return x, z, distribution
