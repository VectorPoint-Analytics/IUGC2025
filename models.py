import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn import functional as F
import timm
from utils import soft_argmax

# --------------------------
# HRNet Encoder Wrapper
# --------------------------

class HRNetEncoderWrapper(nn.Module):
    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()

        # Create HRNet model with feature maps only
        self.backbone = timm.create_model(
            'hrnet_w48', pretrained=pretrained, features_only=True)

        # Modify the first conv layer if input is not 3-channel (e.g., grayscale)
        if in_channels != 3:
            # Access the first conv layer
            old_conv = self.backbone.conv1
            new_conv = nn.Conv2d(in_channels,
                                 out_channels=old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=(old_conv.bias is not None))

            with torch.no_grad():
                if pretrained:
                    new_conv.weight[:] = old_conv.weight.mean(
                        dim=1, keepdim=True)
                    if old_conv.bias is not None:
                        new_conv.bias[:] = old_conv.bias

            self.backbone.conv1 = new_conv

        # Final stage of HRNet-W48 has 480 output channels (high-res path)
        self.out_channels = self.backbone.feature_info[-1]['num_chs']

    def forward(self, x):
        features = self.backbone(x)  # returns a list of feature maps
        return features[-1]  # deepest stage feature map


# --- CBAM Module ---

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# --------------------------
# Heatmap Decoder
# --------------------------

class HRNetWithHeatmapDecoder(nn.Module):
    def __init__(self, hrnet_encoder, num_landmarks=3):
        super(HRNetWithHeatmapDecoder, self).__init__()
        # Expected output shape: [B, 1024, 16, 16]
        self.encoder = hrnet_encoder

        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CBAM(512),

            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=False),  # 32x32
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256),

            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=False),  # 64x64
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAM(128),

            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=False),  # 128x128
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=False),  # 256x256
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=False),  # 512x512
            # Final output: [B, num_landmarks, 512, 512]
            nn.Conv2d(32, num_landmarks, kernel_size=1)
        )

    def forward(self, x):
        features = self.encoder(x)
        heatmaps = self.decoder(features)
        coords = soft_argmax(heatmaps)
        return heatmaps, coords

# --------------------------------
# Spectral PatchGAN Discriminator
# --------------------------------

class SpectralPatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            # Input: [B, 3, H, W]
            spectral_norm(nn.Conv2d(in_channels, base_channels,
                          kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(base_channels, base_channels *
                          2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(
                base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(
                base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 8, 1, kernel_size=4,
                      stride=1, padding=1)  # No SN here
        )

    def forward(self, x):
        return self.net(x)  # Output: [B, 1, H', W']


def main():
    # Verify the input and output shapes
    hrnet_encoder = HRNetEncoderWrapper(in_channels=1, pretrained=False)
    model = HRNetWithHeatmapDecoder(hrnet_encoder, num_landmarks=3)
    input_tensor = torch.randn(1, 1, 512, 512)  # Batch size 1, grayscale image
    heatmaps, coords = model(input_tensor)
    print(f"Heatmaps shape: {heatmaps.shape}")  # Should be [1, 3, 512, 512]
    print(f"Coordinates shape: {coords.shape}")  # Should be [1, 3, 2]

if __name__ == "__main__":
    main()