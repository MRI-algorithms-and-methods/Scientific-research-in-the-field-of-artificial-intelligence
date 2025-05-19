# Standard library
import os
import json
import math
import shutil
import filecmp
from pathlib import Path

# Third-party libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# PyTorch & torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# Custom modules
from utils import (
    get_max_ps_length,
    MRIImageDataset,
    SSIMLoss,
    save_generated_images,
    CombinedLoss,
    CombinedKSpaceRowwiseMSELoss,
)
from models import (
    ConvPhantomEncoder,
    ConvPulseSequenceEncoder,
    MRImageDecoder,
    FusionModule,
)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_per_process_memory_fraction(0.5, 0)
print(f'Using device: {device}')

# Initialize model components
pulse_encoder = ConvPulseSequenceEncoder().to(device)
phantom_encoder = ConvPhantomEncoder().to(device)
fusion = FusionModule().to(device)
decoder = MRImageDecoder().to(device)

# Combine modules into a model container
model = nn.Module()
model.pulse_encoder = pulse_encoder
model.phantom_encoder = phantom_encoder
model.fusion = fusion
model.decoder = decoder
model.to(device)

# Training pipeline
def main():
    subset = ['train', 'test']

    # Define paths
    data_paths = {
        "train": {
            "ps": f'Dataset/{subset[0]}/ps/waveforms',
            "phantom": f'Dataset/{subset[0]}/phantoms',
            "kspace": f'Dataset/{subset[0]}/kspace/sorted',
            "recon": f'Dataset/{subset[0]}/recon/npy',
        },
        "test": {
            "ps": f'Dataset/{subset[1]}/ps/waveforms',
            "phantom": f'Dataset/{subset[1]}/phantoms',
            "kspace": f'Dataset/{subset[1]}/kspace/sorted',
            "recon": f'Dataset/{subset[1]}/recon/npy',
        }
    }

    out_dir = 'out_img/output'
    os.makedirs(out_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    max_ps_length = get_max_ps_length(data_paths["train"]["ps"])
    batch_size = 2

    # Dataloaders
    train_dataset = MRIImageDataset(
        data_paths["train"]["ps"],
        data_paths["train"]["phantom"],
        data_paths["train"]["kspace"],
        data_paths["train"]["recon"],
        transform=transform,
        max_ps_length=max_ps_length
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    test_dataset = MRIImageDataset(
        data_paths["test"]["ps"],
        data_paths["test"]["phantom"],
        data_paths["test"]["kspace"],
        data_paths["test"]["recon"],
        transform=transform,
        max_ps_length=max_ps_length
    )
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=1, pin_memory=True)

    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    writer = SummaryWriter('simulator/runs/mri_model')
    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, ((ps_batch, phantom_batch), target_img) in enumerate(train_dataloader):
            ps_batch = ps_batch.to(device)
            phantom_batch = phantom_batch.to(device)
            target_img = target_img.to(device)

            optimizer.zero_grad()

            pulse_feat = model.pulse_encoder(ps_batch)
            phantom_feat = model.phantom_encoder(phantom_batch)
            fused_feat = model.fusion(phantom_feat, pulse_feat)
            output_img = model.decoder(fused_feat)

            loss = criterion(output_img, target_img)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                avg_loss = running_loss / 10
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {avg_loss:.4f}")
                writer.add_scalar("Loss/train", avg_loss, epoch * len(train_dataloader) + i)
                running_loss = 0.0

        # Save model
        model_path = f'saved_mri_models/model_epoch_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }, model_path)

        scheduler.step()

    writer.close()

if __name__ == '__main__':
    main()
