import torch
from torch import nn


class GAN(nn.Module):
    def __init__(self, patch_size, channels, latent_dim, conditional=False, condition_dim=None):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.latent_layer = torch.nn.Linear(self.latent_dim, 32 * self.patch_size * self.patch_size)
        self.layers = torch.nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, self.channels, kernel_size=3, stride=1, padding=1),
        )
        self.conditional = conditional
        if self.conditional:
            self.condition_layer_1 = torch.nn.Sequential(
                nn.Conv2d(condition_dim, 32, 3, 1, 1),
                nn.ReLU()
            )
            self.condition_layer_2 = torch.nn.Sequential(
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.ReLU()
            )


    def forward(self, z, x=None):
        assert x is not None if self.conditional else True, "Conditional GAN requires conditioning"
        z = self.latent_layer(z)
        z = z.reshape(-1, 32, self.patch_size, self.patch_size)
        if self.conditional:
            x_cond = self.condition_layer_1(x)
            z = torch.cat((z, x_cond), dim=1)
            z = self.condition_layer_2(z)
        return self.layers(z)

    @torch.no_grad
    def sample(self, n, x=None):
        z = torch.randn(n, self.latent_dim, device=self.latent_layer.weight.device)
        return self.forward(z, x)