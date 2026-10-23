"""
STEP 3: MODEL ARCHITECTURE
============================
Two-branch multimodal network:

  Branch A (Spatial):   Pretrained ResNet18 (ImageNet) on the RGB face
                         -> 512-d feature vector
  Branch B (Frequency):  Small CNN on the 1-channel FFT magnitude map
                         -> 128-d feature vector

  Fusion: concatenate [512 + 128] -> fully connected classifier -> 2 classes
          (0 = real, 1 = fake)
"""

import torch
import torch.nn as nn
from torchvision import models


class FrequencyBranch(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),  # 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 64 x 4 x 4
        )
        self.fc = nn.Linear(64 * 4 * 4, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)


class MultimodalDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        super().__init__()

        # Spatial branch: pretrained ResNet18, drop its final FC layer
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.spatial_backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> 512 x 1 x 1
        self.spatial_out_dim = 512

        if freeze_backbone:
            for p in self.spatial_backbone.parameters():
                p.requires_grad = False

        # Frequency branch
        self.freq_branch = FrequencyBranch(out_dim=128)

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.spatial_out_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, rgb, freq):
        spatial_feat = self.spatial_backbone(rgb).flatten(1)   # B x 512
        freq_feat = self.freq_branch(freq)                     # B x 128
        fused = torch.cat([spatial_feat, freq_feat], dim=1)    # B x 640
        out = self.classifier(fused)
        return out


if __name__ == "__main__":
    # quick shape sanity check
    model = MultimodalDeepfakeDetector()
    rgb = torch.randn(2, 3, 224, 224)
    freq = torch.randn(2, 1, 224, 224)
    out = model(rgb, freq)
    print("Output shape:", out.shape)  # expected: torch.Size([2, 2])
