
## DL-based Bloch simulator 

### 📦 `models/`
Defines the core deep learning models:
- `ConvPulseSequenceEncoder`: encodes RF and gradient waveforms.
- `ConvPhantomEncoder`: encodes digital phantom maps.
- `FusionModule`: merges pulse and phantom embeddings.
- `MRImageDecoder`: reconstructs MR images or k-space from fused features.

### 🔧 `utils/`
Utility scripts for:
- Loading and preprocessing data.
- Computing custom losses (e.g., SSIM, MSE, k-space-aware losses).
- Saving output images and intermediate results.

### 🚆 `train.py` / `train_ks.py`
Training scripts for:
- `train.py`: Image reconstruction loss (e.g., MSE/SSIM).
- `train_ks.py`: K-space reconstruction loss (e.g., row-wise MSE).

### 🔍 `test.py` / `test_ks.py`
Evaluation scripts that:
- Load trained models.
- Run inference on test datasets.
- Optionally save generated images or k-space data for inspection.



## Training Configuration

- **Input**:
  - Pulse Sequences
  - Phantoms
- **Output**:
  - MR Image: Shape `(B, 1, 128, 128)` or k-space data

- **Loss**:
  - `CombinedLoss` (image domain)
  - `CombinedKSpaceRowwiseMSELoss` (k-space domain)

- **Optimizer**: Adam  
- **Learning Rate**: `1e-4`  
- **Scheduler**: StepLR (step=10, gamma=0.1)  
- **Epochs**: 50  
- **Batch Size**: 2  



## Output

- Trained models saved to: `saved_mri_models/`
- Reconstructed images saved to: `out_img/output/`
- TensorBoard logs in: `simulator/runs/mri_model`



## Requirements

Install required packages with:

```bash
pip install -r requirements.txt
