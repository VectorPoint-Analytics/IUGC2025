import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import os
import cv2
from utils import AUGMENTATIONS
from models import HRNetEncoderWrapper
from config import MOCO_N_EPOCHS, MOCO_BATCH_SIZE, DEVICE, BASE_LR, HEAD_LR, MOCO_MODEL_LOC

# --------------------------
# MoCo v2 Module
# --------------------------

class MoCoV2(nn.Module):
    def __init__(self, encoder_q, encoder_k, feature_dim=128, K=8192, m=0.999, T=0.2):
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        # 1x1 conv projector to maintain spatial layout
        self.projector_q = nn.Sequential(
            nn.Conv2d(encoder_q.out_channels, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, feature_dim, kernel_size=1)
        )

        self.projector_k = nn.Sequential(
            nn.Conv2d(encoder_k.out_channels, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, feature_dim, kernel_size=1)
        )

        # Freeze key encoder initially
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(feature_dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_flattened):
        # keys_flattened: [B*H*W, C]
        batch = keys_flattened.shape[0]
        ptr = int(self.queue_ptr)
        K = self.K

        if ptr + batch <= K:
            self.queue[:, ptr:ptr + batch] = keys_flattened.T
        else:
            overflow = (ptr + batch) - K
            self.queue[:, ptr:] = keys_flattened[:batch - overflow].T
            self.queue[:, :overflow] = keys_flattened[batch - overflow:].T

        ptr = (ptr + batch) % K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        # Forward through encoders
        q_feat = self.encoder_q(im_q)       # [B, C, H, W]
        k_feat = self.encoder_k(im_k)       # [B, C, H, W]

        # Project
        q = self.projector_q(q_feat)        # [B, C', H, W]
        k = self.projector_k(k_feat)        # [B, C', H, W]

        # Normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        # Flatten spatial locations
        B, C, H, W = q.shape
        q_flat = q.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        k_flat = k.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]

        # Compute positive logits
        l_pos = torch.sum(q_flat * k_flat, dim=-1, keepdim=True)  # [B*H*W, 1]

        # Compute negative logits
        l_neg = torch.mm(q_flat, self.queue.clone().detach())     # [B*H*W, K]

        # Logits: [B*H*W, 1+K]
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(
            logits.shape[0], dtype=torch.long).to(logits.device)

        self._momentum_update_key_encoder()
        self._dequeue_and_enqueue(k_flat)

        return logits, labels


class MoCoDataset(Dataset):
    def __init__(self, image_paths, transform=None, n_views=2):
        """
        Args:
            image_paths: List of image file paths.
            transform: Transform pipeline (e.g., with Albumentations).
            n_views: Number of augmented views to return for each image.
        """
        self.image_paths = image_paths
        self.transform = transform
        self.n_views = n_views

        if not self.image_paths:
            print(
                "No image paths provided. Please check the dataset directory.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = cv2.imread(image_path)

            if image is None:
                print(f"Image at {image_path} could not be read.")
                raise ValueError("Image not readable")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Generate N different augmented views
            views = [self.transform(image=image)['image']
                     for _ in range(self.n_views)]

            return views  # List of tensors of shape [C, H, W]

        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            return self.__getitem__(min(idx + 1, len(self) - 1))


# --------------------------
# Training Loop
# --------------------------

def train_moco(model, dataloader, optimizer, device, epochs=100,
               scheduler=None, save=True, save_loc='moco_v2.pt',
               mixed_precision=True):
    
    model.train()
    model.to(device)
    scaler = GradScaler(enabled=mixed_precision)
    train_losses = []

    base_name, ext = os.path.splitext(save_loc)

    for epoch in range(epochs):
        total_loss = 0.0

        for views in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            try:
                if isinstance(views[0], list):
                    views = [torch.stack(batch_views, dim=0)
                             for batch_views in zip(*views)]

                im_q = views[0].to(device, non_blocking=True).float()
                im_k = views[1].to(device, non_blocking=True).float()

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type=device.type):
                    logits, labels = model(im_q, im_k)
                    loss = F.cross_entropy(logits, labels)

                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

                del im_q, im_k, logits, labels, loss
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("WARNING: OOM error caught. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        if scheduler:
            scheduler.step()

        if save:
            epoch_save_loc = f"{base_name}_epoch{epoch+1}{ext}"
            torch.save(model.state_dict(), epoch_save_loc)
            print(f"Model saved at {epoch_save_loc}")

    return model, train_losses

# --------------------------
# Finetuning Parameters
# --------------------------

def get_finetune_params(model, base_lr=1e-4, head_lr=1e-3):
    param_groups = []

    # Finetune deeper HRNet stages more than shallow ones
    for name, param in model.encoder_q.backbone.named_parameters():
        if not param.requires_grad:
            continue

        if "stages.3" in name:  # deepest stage
            param_groups.append({'params': param, 'lr': base_lr})
        elif "stages.2" in name:
            param_groups.append({'params': param, 'lr': base_lr * 0.5})
        else:
            param_groups.append({'params': param, 'lr': base_lr * 0.1})

    # Add projection head (e.g., model.projector_q)
    param_groups.append(
        {'params': model.projector_q.parameters(), 'lr': head_lr})

    return param_groups


def main():

    # Unlabelled Images Dataset
    unlabelled_dir = "Training/Unlabeled cases"
    image_paths = sorted(glob.glob(os.path.join(unlabelled_dir, "Example*.jpg")))
    print(f"Found {len(image_paths)} unlabelled standard plane images.")

    dataset = MoCoDataset(image_paths, n_views=2, transform=AUGMENTATIONS)
    dataloader = DataLoader(dataset, batch_size=MOCO_BATCH_SIZE, shuffle=True)

    # Initialize encoders
    encoder_q = HRNetEncoderWrapper(in_channels=1, pretrained=True)
    encoder_k = HRNetEncoderWrapper(in_channels=1, pretrained=True)

    # Create MoCo model
    moco_model = MoCoV2(encoder_q, encoder_k)

    # Freeze the key encoder initially
    for param in moco_model.encoder_k.parameters():
        param.requires_grad = False

    param_groups = get_finetune_params(moco_model, base_lr=BASE_LR, head_lr=HEAD_LR)

    # Define optimizer
    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MOCO_N_EPOCHS)

    # Train the model
    train_moco(moco_model, dataloader, optimizer, DEVICE, epochs=MOCO_N_EPOCHS, save=True,
               save_loc=MOCO_MODEL_LOC, scheduler=scheduler, mixed_precision=True)

if __name__ == "__main__":
    main()