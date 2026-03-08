import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network (Dong et al., 2014).

    Input : Bicubic-upsampled LR image — same spatial dimensions as the HR target.
    Output: SR image of identical spatial size.

    Architecture
    ------------
    1. Feature extraction  : Conv2d(3,  64, 9x9) + ReLU
    2. Non-linear mapping  : Conv2d(64, 32, 1x1) + ReLU
    3. Reconstruction      : Conv2d(32,  3, 5x5)
    """

    def __init__(self, num_channels: int = 3):
        super().__init__()
        self.feature_extraction  = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.non_linear_mapping  = nn.Conv2d(64,           32, kernel_size=1, padding=0)
        self.reconstruction      = nn.Conv2d(32, num_channels,  kernel_size=5, padding=2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.feature_extraction(x))
        x = F.relu(self.non_linear_mapping(x))
        return self.reconstruction(x)


class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Neural Network (Shi et al., 2016).

    Input : LR image at its *native* low resolution.
    Output: HR image via learned sub-pixel convolution (pixel shuffle).

    Architecture
    ------------
    1. Conv2d(3,  64, 5x5) + ReLU
    2. Conv2d(64, 32, 3x3) + ReLU
    3. Conv2d(32, C * r^2, 3x3)   where C = num_channels, r = scale_factor
    4. PixelShuffle(r)
    """

    def __init__(self, scale_factor: int = 4, num_channels: int = 3):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return torch.clamp(x, 0.0, 1.0)


def build_model(architecture: str = 'espcn', scale_factor: int = 2, num_channels: int = 3) -> nn.Module:
    """Factory that returns the requested super-resolution model."""
    arch = architecture.lower()
    if arch == 'srcnn':
        return SRCNN(num_channels=num_channels)
    elif arch == 'espcn':
        return ESPCN(scale_factor=scale_factor, num_channels=num_channels)
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose 'srcnn' or 'espcn'.")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    srcnn = build_model('srcnn', scale_factor=2).to(device)
    x1 = torch.randn(1, 3, 256, 256).to(device)
    print(f"SRCNN  | Input: {tuple(x1.shape)}  ->  Output: {tuple(srcnn(x1).shape)}")
    print(f"       | Parameters: {sum(p.numel() for p in srcnn.parameters()):,}\n")

    espcn = build_model('espcn', scale_factor=2).to(device)
    x2 = torch.randn(1, 3, 32, 32).to(device)
    print(f"ESPCN  | Input: {tuple(x2.shape)}  ->  Output: {tuple(espcn(x2).shape)}")
    print(f"       | Parameters: {sum(p.numel() for p in espcn.parameters()):,}")
