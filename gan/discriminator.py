import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, patch_size, channels, conditional=False, condition_dim=None):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels

        self.conv_layers = torch.nn.Sequential(
            nn.Conv2d(self.channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Calculate the size of the flattened features after Conv2d
        output_features = 16 * patch_size * patch_size

        self.fc_layers = torch.nn.Sequential(
            nn.Linear(output_features, 1),
            nn.Sigmoid()
        )
        self.conditional = conditional
        if conditional:
            self.condition_layer = torch.nn.Sequential(
                nn.Conv2d(self.channels + condition_dim, self.channels, 3, 1, 1),
                nn.ReLU()
            )

    def forward(self, y, x=None):
        if self.conditional:
            y = torch.cat((y, x), dim=1)
            y = self.condition_layer(y)
        y = self.conv_layers(y)
        y = torch.flatten(y, 1)

        return self.fc_layers(y)
