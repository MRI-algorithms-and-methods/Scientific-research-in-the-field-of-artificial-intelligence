import torch
import torch.nn as nn
import torch.nn.functional as F



# --- Convolutional PulseSequenceEncoder ---
class ConvPulseSequenceEncoder(nn.Module):
    def __init__(self, input_channels=7, model_dim=512, output_dim=512):
        super(ConvPulseSequenceEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, model_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(model_dim, model_dim * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(model_dim * 2, model_dim * 4, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(model_dim * 4, output_dim)

    def forward(self, x):  # x: [B, 7, 11024]
        x = x.unsqueeze(2)  # [B, 7, 1, 11024]
        x = self.activation(self.conv1(x))  # [B, model_dim, 1, 11024]
        x = self.activation(self.conv2(x))  # [B, model_dim*2, 1, 11024]
        x = self.activation(self.conv3(x))  # [B, model_dim*4, 1, 11024]
        x = x.squeeze(2)  # [B, model_dim*4, 11024]
        x = x.permute(0, 2, 1)  # [B, 11024, model_dim*4]
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  # [B, model_dim*4]
        return self.output_proj(x)  # [B, output_dim]

# --- Convolutional PhantomEncoder ---
class ConvPhantomEncoder(nn.Module):
    def __init__(self, input_channels=5, model_dim=64, output_dim=512):
        super(ConvPhantomEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, model_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(model_dim, model_dim * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(model_dim * 2, model_dim * 4, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(model_dim * 4, output_dim)

    def forward(self, x):  # x: [B, 5, 250, 250]
        # Assuming x is of shape [2, 250, 250, 5]
        x = x.permute(0, 3, 1, 2)  # Now x is [2, 5, 250, 250], suitable for Conv2d
        x = self.activation(self.conv1(x))  # [B, model_dim, 250, 250]
        x = self.activation(self.conv2(x))  # [B, model_dim*2, 250, 250]
        x = self.activation(self.conv3(x))  # [B, model_dim*4, 250, 250]
        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)  # [B, 62500, model_dim*4]
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  # [B, model_dim*4]
        return self.output_proj(x)  # [B, output_dim]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, model_dim)
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x



# --- FusionModule ---
class FusionModule(nn.Module):
    def __init__(self, input_dim=1024, output_dim=512):
        super(FusionModule, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, img_feat, num_feat):  # both: [B, 512]
        fused = torch.cat((img_feat, num_feat), dim=1)  # [B, 1024]
        return self.fc(fused)  # [B, 512]

class FusionModuleWithPS_param(nn.Module):
    def __init__(self, phantom_feat_dim=512, ps_feature_dim=512, ps_param_dim=10, hidden_dim=512, output_dim=512):
        super(FusionModule, self).__init__()
        self.param_proj = nn.Linear(ps_param_dim, hidden_dim)
        self.fc = nn.Linear(phantom_feat_dim + ps_feature_dim + hidden_dim, output_dim)

    def forward(self, phantom_feat, ps_feature, ps_params):  # все [B, ...]
        param_feat = self.param_proj(ps_params)  # [B, hidden_dim]
        fused = torch.cat((phantom_feat, ps_feature, param_feat), dim=1)  # [B, 512 + 512 + hidden_dim]
        return self.fc(fused)  # [B, 512]

# --- MRImageDecoder ---
class MRImageDecoder(nn.Module):
    def __init__(self, input_dim=512, base_channels=64, output_channels=1):
        super(MRImageDecoder, self).__init__()

        self.fc = nn.Linear(input_dim, base_channels * 8 * 8 * 8)  # project to 8×8 feature map
        self.relu = nn.ReLU()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),  # 8×8 → 16×16
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),  # 16×16 → 32×32
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),  # 32×32 → 64×64
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),

            nn.ConvTranspose2d(base_channels, output_channels, kernel_size=4, stride=2, padding=1),  # 64×64 → 128×128
            nn.Tanh()  # assuming MR image values are normalized to [-1, 1]
        )

    def forward(self, x):  # x: [B, 512]
        x = self.relu(self.fc(x))  # [B, 64*8*8*8]
        x = x.view(-1, 512, 8, 8)  # reshape to [B, C, H, W]
        x = self.decoder(x)  # [B, 1, 128, 128]
        return x


class MR_kspaceDecoder(nn.Module):
    def __init__(self, input_dim=512, base_channels=64, output_channels=2):
        super(MR_kspaceDecoder, self).__init__()

        self.fc = nn.Linear(input_dim, base_channels * 8 * 8 * 8)  # project to 8×8 feature map
        self.relu = nn.ReLU()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),  # 8×8 → 16×16
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),  # 16×16 → 32×32
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),  # 32×32 → 64×64
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),

            nn.ConvTranspose2d(base_channels, output_channels, kernel_size=4, stride=2, padding=1),  # 64×64 → 128×128
            nn.Tanh()  # assuming both real and imag channels are normalized to [-1, 1]
        )

    def forward(self, x):  # x: [B, 512]
        x = self.relu(self.fc(x))  # [B, base_channels * 8 * 8]
        x = x.view(-1, 512, 8, 8)  # reshape to [B, C, H, W]
        x = self.decoder(x)  # [B, 2, 128, 128]
        return x
