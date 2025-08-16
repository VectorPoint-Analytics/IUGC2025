import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from utils import AUGMENTATIONS,  \
    entropy_loss, freeze_encoder, unfreeze_encoder, unfreeze_encoder_stage2
from models import HRNetEncoderWrapper, HRNetWithHeatmapDecoder, SpectralPatchGANDiscriminator
from config import FINETUNING_N_EPOCHS, FINETUNING_BATCH_SIZE, DEVICE, \
    MOCO_MODEL_LOC, G_LEARNING_RATE, D_LEARNING_RATE, UNFREEZE_EPOCHS, UNFREEZE_EPOCHS2, \
    GENERATOR_LOC, DISCRIMINATOR_LOC
from moco_v2 import MoCoV2

from heatmap_dataset import HeatmapLandmarkDataset

# --------------------------
# Dataset Preparation
# --------------------------

labelled_dataset = HeatmapLandmarkDataset(
    csv_file='Training/Labeled cases/label.csv',
    img_dir='Training/Labeled cases',
    transform=AUGMENTATIONS,
    num_views=3
)

torch.manual_seed(42) # For reproducibility

# Split dataset into training and validation sets
train_size = int(0.8 * len(labelled_dataset))
val_size = len(labelled_dataset) - train_size
train_dataset, val_dataset = random_split(
    labelled_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=FINETUNING_BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=FINETUNING_BATCH_SIZE,
                        shuffle=False, num_workers=4, pin_memory=True)


# --------------------------
# Initialization
# --------------------------

# Loss functions
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()

# Mixed precision
scaler_G = GradScaler()
scaler_D = GradScaler()

# Models
encoder_q = HRNetEncoderWrapper(pretrained=False)
encoder_k = HRNetEncoderWrapper(pretrained=False)

moco_model = MoCoV2(encoder_q, encoder_k, K=8192)

moco_model.load_state_dict(torch.load(MOCO_MODEL_LOC, map_location=DEVICE), strict=False)
encoder = encoder_q
generator = HRNetWithHeatmapDecoder(encoder).to(DEVICE)
discriminator = SpectralPatchGANDiscriminator(
    in_channels=3, base_channels=64).to(DEVICE)

# Optimizers
optimizer_G = torch.optim.AdamW(
    generator.parameters(), lr=G_LEARNING_RATE, weight_decay=1e-4)
optimizer_D = torch.optim.AdamW(
    discriminator.parameters(), lr=D_LEARNING_RATE, weight_decay=1e-4)

# scheduler
scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_G, T_max=FINETUNING_N_EPOCHS, eta_min=5e-6)
scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_D, T_max=FINETUNING_N_EPOCHS, eta_min=1e-7)

# --------------------------
# Training and validation functions
# --------------------------

def train_epoch(generator, discriminator,
                dataloader, optimizer_G, optimizer_D,
                scaler_G, scaler_D,
                DEVICE, lambda_adv=0.01):

    generator.train()
    discriminator.train()

    total = {'heatmap': 0, 'coord': 0, 'adv_G': 0, 'adv_D': 0}
    num_batches = len(dataloader)

    for images, gt_heatmaps, gt_coords in tqdm(dataloader, desc="Training"):
        images = images.to(DEVICE)
        gt_heatmaps = gt_heatmaps.to(DEVICE)
        gt_coords = gt_coords.to(DEVICE)

        # === Generator forward ===
        with autocast(DEVICE_type=DEVICE.type):
            pred_heatmaps, pred_coords = generator(images)
            pred_coords = pred_coords.reshape(-1, 6)

            # Losses
            loss_heatmap = mse_loss(pred_heatmaps, gt_heatmaps)
            loss_coords = mse_loss(pred_coords, gt_coords)

            # Adversarial loss (G tries to fool D)
            pred_D_fake = discriminator(pred_heatmaps)
            adv_loss_G = bce_loss(pred_D_fake, torch.ones_like(pred_D_fake))

            entropy_penalty = entropy_loss(pred_heatmaps)
            loss_G = loss_heatmap + 10 * loss_coords + \
                lambda_adv * adv_loss_G + 0.001 * entropy_penalty

        # === Generator Backward ===
        optimizer_G.zero_grad()
        scaler_G.scale(loss_G).backward()
        scaler_G.step(optimizer_G)
        scaler_G.update()

        # === Discriminator forward ===
        with autocast(DEVICE_type=DEVICE.type):
            pred_D_fake = discriminator(pred_heatmaps.detach())
            pred_D_real = discriminator(gt_heatmaps)

            adv_loss_D_real = bce_loss(
                pred_D_real, torch.ones_like(pred_D_real))
            adv_loss_D_fake = bce_loss(
                pred_D_fake, torch.zeros_like(pred_D_fake))
            adv_loss_D = (adv_loss_D_real + adv_loss_D_fake) / 2

        # === Discriminator Backward ===
        optimizer_D.zero_grad()
        scaler_D.scale(adv_loss_D).backward()
        scaler_D.step(optimizer_D)
        scaler_D.update()

        # === Logging ===
        total['heatmap'] += loss_heatmap.item()
        total['coord'] += loss_coords.item()
        total['adv_G'] += adv_loss_G.item()
        total['adv_D'] += adv_loss_D.item()

        # === Clean up memory ===
        del images, gt_heatmaps, gt_coords
        del pred_heatmaps, pred_coords
        del pred_D_fake, pred_D_real
        del loss_heatmap, loss_coords, adv_loss_G, adv_loss_D, adv_loss_D_real, adv_loss_D_fake, loss_G
        del entropy_penalty
        torch.cuda.empty_cache()

    return {k: v / num_batches for k, v in total.items()}


# === Validation Epoch ===
@torch.no_grad()
def validate_epoch(generator, discriminator, dataloader, DEVICE, lambda_adv=0.01):
    generator.eval()
    discriminator.eval()

    total = {'heatmap': 0, 'coord': 0, 'adv_G': 0, 'adv_D': 0}
    num_batches = len(dataloader)

    for images, gt_heatmaps, gt_coords in tqdm(dataloader, desc="Validation"):
        images = images.to(DEVICE)
        gt_heatmaps = gt_heatmaps.to(DEVICE)
        gt_coords = gt_coords.to(DEVICE)

        with autocast(DEVICE_type=DEVICE.type):
            pred_heatmaps, pred_coords = generator(images)
            pred_coords = pred_coords.reshape(-1, 6)

            loss_heatmap = mse_loss(pred_heatmaps, gt_heatmaps)
            loss_coords = mse_loss(pred_coords, gt_coords)

            # Adversarial losses
            pred_D_fake = discriminator(pred_heatmaps)
            pred_D_real = discriminator(gt_heatmaps)

            adv_loss_G = bce_loss(pred_D_fake, torch.ones_like(pred_D_fake))
            adv_loss_D_real = bce_loss(
                pred_D_real, torch.ones_like(pred_D_real))
            adv_loss_D_fake = bce_loss(
                pred_D_fake, torch.zeros_like(pred_D_fake))
            adv_loss_D = (adv_loss_D_real + adv_loss_D_fake) / 2

        # Logging
        total['heatmap'] += loss_heatmap.item()
        total['coord'] += loss_coords.item()
        total['adv_G'] += adv_loss_G.item()
        total['adv_D'] += adv_loss_D.item()

        # === Clean up memory ===
        del images, gt_heatmaps, gt_coords
        del pred_heatmaps, pred_coords
        del pred_D_fake, pred_D_real
        del loss_heatmap, loss_coords, adv_loss_G, adv_loss_D, adv_loss_D_real, adv_loss_D_fake
        torch.cuda.empty_cache()

    return {k: v / num_batches for k, v in total.items()}

def main():

    train_losses = []
    val_losses = []

    best_val_coord_loss = float('inf')

    freeze_encoder(generator)

    for epoch in range(FINETUNING_N_EPOCHS):

        if epoch == UNFREEZE_EPOCHS:
            unfreeze_encoder(generator)
            print(f"Unfreezing Encoder stage-3")

        if epoch == UNFREEZE_EPOCHS2:
            unfreeze_encoder_stage2(generator)
            print(f"Unfreezing Encoder stage-2")

        print(f"\n======= Epoch {epoch+1}/{FINETUNING_N_EPOCHS} =======\n")

        train_logs = train_epoch(generator, discriminator, train_loader,
                                optimizer_G, optimizer_D,
                                scaler_G, scaler_D,
                                DEVICE)

        val_logs = validate_epoch(generator, discriminator, val_loader, DEVICE)

        scheduler_G.step()
        scheduler_D.step()

        train_losses.append(train_logs)
        val_losses.append(val_logs)

        print(f"[Train] Heatmap: {train_logs['heatmap']:.6f} | Coord: {train_logs['coord']:.6f} | Adv_G: {train_logs['adv_G']:.6f} | Adv_D: {train_logs['adv_D']:.6f}")
        print(
            f"[ Val ] Heatmap: {val_logs['heatmap']:.6f} | Coord: {val_logs['coord']:.6f} | Adv_G: {val_logs['adv_G']:.6f} | Adv_D: {val_logs['adv_D']:.6f}")

        # Save best model
        if val_logs['coord'] < best_val_coord_loss:
            best_val_coord_loss = val_logs['coord']
            torch.save(generator.state_dict(), GENERATOR_LOC)
            torch.save(discriminator.state_dict(), DISCRIMINATOR_LOC)
            print(
                f"🔽 Best model saved (val coord loss = {best_val_coord_loss:.6f})")

if __name__ == "__main__":
    main()