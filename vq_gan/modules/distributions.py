class GaussianDistribution(nn.Module):

    def __init__(self, parameters: torch.Tensor) -> None:
        super().__init__()

        self.mean, self.log_variance = parameters.chunk(chunks=2, dim=1)
    
    def sample(self) -> torch.Tensor:

        epsilon = torch.randn_like(self.mean, device=self.mean.device)
        standard_deviation = torch.exp(0.5 * self.log_variance)
        x = epsilon * standard_deviation + self.mean

        return x
    
    @property
    def divergence(self) -> torch.Tensor:

        x = self.log_variance - self.log_variance.exp() - self.mean + 1
        x = x.mean() * -0.5

        return x
