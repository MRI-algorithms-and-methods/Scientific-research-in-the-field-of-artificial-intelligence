import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import h5py
from torchvision import transforms

from utils import (
    MRIImageDataset,
    SSIMLoss,
    save_generated_images,
    CombinedLoss,
    CombinedKSpaceRowwiseMSELoss,
)
from models import (
    ConvPhantomEncoder,
    ConvPulseSequenceEncoder,
    MR_kspaceDecoder,
    FusionModule,
)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_per_process_memory_fraction(0.5, 0)  # Adjust memory usage
print(f'Using device: {device}')

# Initialize model components
pulse_encoder = ConvPulseSequenceEncoder().to(device)
phantom_encoder = ConvPhantomEncoder().to(device)
fusion = FusionModule().to(device)
decoder = MR_kspaceDecoder().to(device)

# Combine all modules into a single model container
model = nn.Module()
model.pulse_encoder = pulse_encoder
model.phantom_encoder = phantom_encoder
model.fusion = fusion
model.decoder = decoder
model.to(device)



# Dataset setup
def main():
    subset = ['train', 'test']  # Train, Validation, Test subsets
    contrast = 'T1'

    # Training data
    ps_dir_tr = fr'Dataset/{subset[0]}/ps/waveforms'
    phantom_dir_tr = fr'Dataset/{subset[0]}/phantoms'
    ks_dir_tr = fr'Dataset/{subset[0]}/kspace/sorted'
    recon_img_dir_tr = fr'Dataset/{subset[0]}/recon/npy'

    # Test data
    ps_dir_ts = fr'Dataset/{subset[1]}/ps/waveforms'
    phantom_dir_ts = fr'Dataset/{subset[1]}/phantoms'
    ks_dir_ts = fr'Dataset/{subset[1]}/kspace/sorted'
    recon_img_dir_ts = fr'Dataset/{subset[1]}/recon/npy'

    out_dir = fr'out_img\output/output'
    os.makedirs(out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        ])

    max_ps_length = get_max_ps_length(ps_dir_tr)

    # Training dataset and dataloader
    batch_size = 4
    train_dataset = MRIImageDataset(ps_dir_tr, phantom_dir_tr, ks_dir_tr, recon_img_dir_tr, transform=transform, max_ps_length=max_ps_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # Test dataset and dataloader
    test_dataset = MRIImageDataset(ps_dir_ts, phantom_dir_ts, ks_dir_ts, recon_img_dir_ts, transform=transform, max_ps_length=max_ps_length)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=1, pin_memory=True)

    criterion = nn.MSELoss()
    # criterion = SSIMLoss().to(device)  # where device is torch.device('cuda') or 'cpu'
    # criterion = CombinedKSpaceRowwiseMSELoss(central_fraction=0.5)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    writer = SummaryWriter('simulator/runs/mri_model')
    epochs = 30

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for i, ((ps_batch, phantom_batch), (ks_real_batch, ks_imag_batch)) in enumerate(train_dataloader):
            ps_batch = ps_batch.to(device)  # shape: [B, 7, 11021]
            phantom_batch = phantom_batch.to(device)  # shape: [B, 5, 250, 250]
            ks_real_batch = ks_real_batch.to(device)
            ks_imag_batch = ks_imag_batch.to(device)
            # Combine real and imag into a 2-channel tensor
            ks_batch = torch.cat([ks_real_batch, ks_imag_batch], dim=1)  # shape: [B, 2, 128, 128]

            # plot_ps(ps_batch[0].cpu())
            # plot_phantom(phantom_batch[0].cpu())
            # im = np.squeeze(target_img[0].cpu())
            # plot_ks_im(im, im, im)


            optimizer.zero_grad()

            # Forward pass through all components
            pulse_feat = model.pulse_encoder(ps_batch)
            phantom_feat = model.phantom_encoder(phantom_batch)
            fused_feat = model.fusion(phantom_feat, pulse_feat)
            output_ks = model.decoder(fused_feat)

            loss = criterion(output_ks, ks_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                avg_loss = running_loss / 10
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {avg_loss:.4f}")
                writer.add_scalar("Loss/train", avg_loss, epoch * len(train_dataloader) + i)
                running_loss = 0.0

        # Validation phase (using test data)
        model_name = fr'saved_mri_models\model_epoch_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }, model_name)
        #
        # with torch.no_grad():
        #     val_loss = 0.0
        #     # for ps_batch, phantom_batch, target_img in test_dataloader:
        #     for  i, ((ps_batch, phantom_batch), (ks_real_batch, ks_imag_batch)) in enumerate(test_dataloader):
        #         ps_batch = ps_batch.to(device)
        #         phantom_batch = phantom_batch.to(device)
        #         ks_real_batch = ks_real_batch.to(device)
        #         ks_imag_batch = ks_imag_batch.to(device)
        #         # Combine real and imag into a 2-channel tensor
        #         ks_batch = torch.cat([ks_real_batch, ks_imag_batch], dim=1)  # shape: [B, 2, 128, 128]
        #
        #
        #         pulse_feat = model.pulse_encoder(ps_batch)
        #         phantom_feat = model.phantom_encoder(phantom_batch)
        #         fused_feat = model.fusion(phantom_feat, pulse_feat)
        #         output_ks = model.decoder(fused_feat)
        #
        #         loss = criterion(output_ks, ks_batch)
        #         val_loss += loss.item()
        #
        #     avg_val_loss = val_loss / len(test_dataloader)
        #     print(f"Validation Loss after Epoch [{epoch + 1}/{epochs}]: {avg_val_loss:.4f}")
        #     writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        #
        # # Save generated images for inspection
        # if (epoch + 1) % 1 == 0:  # Modify to save more frequently if needed
        #     save_generated_images_from_ks(out_dir, output_ks, ks_batch, epoch)

        scheduler.step()

    writer.close()

if __name__ == '__main__':
    main()
