import torch
import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------
# Soft Argmax based coordinate extraction
# ---------------------------

def soft_argmax(heatmaps, temperature=10.0):
    B, C, H, W = heatmaps.shape
    heatmaps = F.softmax(heatmaps.view(B, C, -1) *
                         temperature, dim=-1).view(B, C, H, W)

    # Create coordinate grid
    xs = torch.linspace(0, W - 1, W, device=heatmaps.device)
    ys = torch.linspace(0, H - 1, H, device=heatmaps.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

    exp_x = torch.sum(heatmaps * grid_x[None, None, :, :], dim=(2, 3))
    exp_y = torch.sum(heatmaps * grid_y[None, None, :, :], dim=(2, 3))
    coords = torch.stack([exp_x, exp_y], dim=-1)  # [B, C, 2]

    # Normalize
    coords[..., 0] /= (W - 1)
    coords[..., 1] /= (H - 1)

    return coords

# --------------------------------
# Hard Argmax based coordinate extraction
# --------------------------------

def get_coordinates_from_heatmaps(heatmaps):
    """
    Args:
        heatmaps: [B, C, H, W] tensor of heatmaps.
    
    Returns:
        coords: [B, C, 2] tensor of (x, y) coordinates in heatmap space.
    """
    B, C, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(B, C, -1)
    max_vals, idx = heatmaps_reshaped.max(dim=-1)

    # Get coarse predictions
    x = idx % W
    y = idx // W
    coords = torch.stack((x, y), dim=-1).float()

    # Optional sub-pixel refinement
    for b in range(B):
        for c in range(C):
            px = int(coords[b, c, 0])
            py = int(coords[b, c, 1])
            if 1 < px < W - 1 and 1 < py < H - 1:
                dx = (
                    heatmaps[b, c, py, px + 1]
                    - heatmaps[b, c, py, px - 1]
                ).item()
                dy = (
                    heatmaps[b, c, py + 1, px]
                    - heatmaps[b, c, py - 1, px]
                ).item()
                coords[b, c, 0] += 0.25 * dx
                coords[b, c, 1] += 0.25 * dy

    return coords


# --------------------------------
# Image Transformations
# --------------------------------

class AddSpeckleNoise(A.ImageOnlyTransform):

    """
    Add speckle noise to the image.
    Speckle noise is multiplicative noise that can be added to images.
    It is often used in ultrasound imaging to simulate realistic noise conditions.
    """

    def __init__(self, mean=0.0, std=0.01, always_apply=False, p=0.5):
        super(AddSpeckleNoise, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std

    def apply(self, image, **params):
        noise = np.random.normal(
            self.mean, self.std, image.shape).astype(np.float32)
        noisy_image = image + image * noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)


AUGMENTATIONS = A.Compose([
    A.OneOf([
        A.ElasticTransform(alpha=5, sigma=10, alpha_affine=10, border_mode=0)
    ], p=0.8),

    A.OneOf([
        A.RandomBrightnessContrast(),
        A.GaussianBlur(blur_limit=(3, 5)),
        AddSpeckleNoise(mean=0.0, std=0.01)
    ], p=0.5),

    A.Normalize(mean=[0.0], std=[255.0]),
    ToTensorV2()
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


# -------------------------------- 
# Finetuning Utilities
# --------------------------------

def freeze_encoder(model):
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model):
    for name, param in model.encoder.backbone.named_parameters():
        if "stage3" in name:
            param.requires_grad = True


def unfreeze_encoder_stage2(model):
    for name, param in model.encoder.backbone.named_parameters():
        if "stage2" in name:
            param.requires_grad = True


def entropy_loss(heatmaps):
    B, C, H, W = heatmaps.shape
    prob = F.softmax(heatmaps.view(B, C, -1), dim=-1)
    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)
    return entropy.mean()
