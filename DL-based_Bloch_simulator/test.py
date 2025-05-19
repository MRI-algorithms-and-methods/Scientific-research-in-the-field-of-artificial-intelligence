import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from models import ConvPulseSequenceEncoder, ConvPhantomEncoder, MRImageDecoder, FusionModule
from utils import MRIImageDataset, save_generated_images_from_ks, get_max_ps_length, save_generated_images

# ---------------------- Setup ---------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inference using device: {device}")

# ---------------------- Model Setup ---------------------- #
# Instantiate model components
pulse_encoder = ConvPulseSequenceEncoder().to(device)
phantom_encoder = ConvPhantomEncoder().to(device)
fusion = FusionModule().to(device)
decoder = MRImageDecoder().to(device)

# Combine into one model container
model = torch.nn.Module()
model.pulse_encoder = pulse_encoder
model.phantom_encoder = phantom_encoder
model.fusion = fusion
model.decoder = decoder
model.to(device)

# ---------------------- Load Weights ---------------------- #
checkpoint_path = 'saved_mri_models/model_epoch_9.pth'  # Modify as needed
checkpoint = torch.load(checkpoint_path, map_location=device)
model_dict = model.state_dict()

# Filter out unnecessary keys
pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and model_dict[k].size() == v.size()}

# Overwrite entries in the existing state dict
model_dict.update(pretrained_dict)

# Load the new state dict
model.load_state_dict(model_dict)

model.eval()
print(f"Loaded model from {checkpoint_path}")

# ---------------------- Data Setup ---------------------- #
# Paths (modify as needed)
subset = 'test'

ps_dir = f'Dataset/{subset}/ps/waveforms'
phantom_dir = f'Dataset/{subset}/phantoms'
ks_real_dir = f'Dataset/{subset}/kspace/sorted'
ks_imag_dir = f'Dataset/{subset}/kspace/sorted'  # If imag is stored together, modify if needed
recon_img_dir = f'Dataset/{subset}/recon/npy'

# Transform & Dataset
transform = transforms.Compose([transforms.ToTensor()])
max_ps_length = get_max_ps_length(ps_dir)

inference_dataset = MRIImageDataset(ps_dir, phantom_dir, ks_real_dir, recon_img_dir, transform=transform, max_ps_length=max_ps_length)
inference_loader = DataLoader(inference_dataset, batch_size=4, shuffle=True)

# Output dir
output_dir = 'inference_outputs'
os.makedirs(output_dir, exist_ok=True)

# ---------------------- Inference Loop ---------------------- #
with torch.no_grad():
    for idx, ((ps_batch, phantom_batch), (target_img)) in enumerate(inference_loader):
        ps_batch = ps_batch.to(device)  # shape: [B, 7, 11021]
        phantom_batch = phantom_batch.to(device)  # shape: [B, 5, 250, 250]
        target_img = target_img.to(device)  # shape: [B, 1, 128, 128]

        # Run model
        pulse_feat = model.pulse_encoder(ps_batch)
        phantom_feat = model.phantom_encoder(phantom_batch)
        fused_feat = model.fusion(phantom_feat, pulse_feat)
        im_pred = model.decoder(fused_feat)
        print(im_pred.shape, target_img.shape)

        # Visualize prediction vs ground truth
        save_generated_images(output_dir, im_pred, target_img, epoch=idx)

        print(f"Saved output for sample {idx + 1}")

print("Inference complete.")
